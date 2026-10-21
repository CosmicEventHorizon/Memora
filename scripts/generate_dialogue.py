import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from memora.acoustic import AcousticPredictor
from memora.codec import DualTokenizer
from memora.decoder import AssistantDecoder
from memora.dialogue import CONTROL_TOKENS, reply_logits
from memora.paths import MODEL_DIR, RUN_DIR
from memora.semantic import MemoraLanguageModel
from memora.tts import load_voice, synthesize


def sample(logits, generator, temperature, top_k):
    logits = logits.float() / temperature
    values, indices = logits.topk(min(top_k, logits.numel()))
    selected = torch.multinomial(values.softmax(-1), 1, generator=generator)
    return int(indices[selected].item())


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Hello, how are you?")
    parser.add_argument("--prompt-audio", type=Path)
    parser.add_argument("--prompt-voice", type=Path)
    parser.add_argument("--maximum-response-frames", type=int, default=128)
    parser.add_argument("--minimum-response-frames", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument(
        "--semantic-checkpoint",
        type=Path,
        default=MODEL_DIR / "semantic" / "best.pt",
    )
    parser.add_argument(
        "--acoustic-checkpoint",
        type=Path,
        default=MODEL_DIR / "acoustic" / "best.pt",
    )
    parser.add_argument(
        "--decoder-checkpoint",
        type=Path,
        default=MODEL_DIR / "decoder" / "best.pt",
    )
    parser.add_argument("--output", type=Path, default=RUN_DIR / "conversation")
    args = parser.parse_args()

    shutil.rmtree(args.output, ignore_errors=True)
    args.output.mkdir(parents=True)
    if args.prompt_audio:
        prompt_wave, sample_rate = sf.read(args.prompt_audio, dtype="float32")
        if prompt_wave.ndim == 2:
            prompt_wave = prompt_wave.mean(axis=1)
        if sample_rate != 24_000:
            prompt_wave = resample_poly(prompt_wave, 24_000, sample_rate).astype(np.float32)
    else:
        if args.prompt_voice is None:
            parser.error("provide --prompt-audio or --prompt-voice")
        prompt_wave = synthesize(load_voice(args.prompt_voice), args.prompt).astype(np.float32)
    prompt_wave = np.pad(prompt_wave, (0, int(0.5 * 24000)))
    sf.write(args.output / "prompt.wav", prompt_wave, 24000)

    frame_samples = 1920
    frames = (len(prompt_wave) + frame_samples - 1) // frame_samples
    padded = np.pad(prompt_wave, (0, frames * frame_samples - len(prompt_wave)))
    codec = DualTokenizer()
    prompt_semantic, _ = codec.encode(torch.from_numpy(padded)[None, None].cuda())
    prompt_semantic = prompt_semantic[0, :frames].long()
    del codec
    torch.cuda.empty_cache()

    prefix = torch.cat(
        (
            torch.tensor([CONTROL_TOKENS["<user>"]], device="cuda"),
            prompt_semantic,
            torch.tensor(
                [CONTROL_TOKENS["<end_turn>"], CONTROL_TOKENS["<assistant>"]],
                device="cuda",
            ),
        )
    )

    package = torch.load(args.semantic_checkpoint, map_location="cuda", weights_only=True)
    semantic_model = MemoraLanguageModel.from_package(package, "cuda").to(torch.bfloat16).eval()
    generator = torch.Generator(device="cuda").manual_seed(42)
    sequence = prefix
    response = []
    ended = False
    for step in range(args.maximum_response_frames):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = semantic_model(sequence[None])["logits"][0, -1]
        logits = reply_logits(logits)
        if step < args.minimum_response_frames:
            logits[CONTROL_TOKENS["<end_turn>"]] = float("-inf")
        token = sample(logits, generator, args.temperature, args.top_k)
        if token == CONTROL_TOKENS["<end_turn>"]:
            ended = True
            break
        if token >= 2048:
            raise RuntimeError(f"Unexpected control token in assistant audio: {token}")
        response.append(token)
        sequence = torch.cat((sequence, torch.tensor([token], device="cuda")))
    if not response:
        raise RuntimeError("Semantic model generated an empty response")
    response_semantic = torch.tensor(response, dtype=torch.long, device="cuda")
    del semantic_model, package, sequence
    torch.cuda.empty_cache()

    acoustic_checkpoint = args.acoustic_checkpoint
    package = torch.load(acoustic_checkpoint, map_location="cuda", weights_only=True)
    acoustic_model = AcousticPredictor.from_package(package, "cuda").to(torch.bfloat16).eval()
    history = torch.empty(7, 0, dtype=torch.long, device="cuda")
    generated = []
    for step in range(len(response)):
        start = max(0, step + 1 - acoustic_model.config.context)
        semantic_context = response_semantic[start : step + 1][None]
        acoustic_context = history[:, start:step][None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            frame = acoustic_model.generate_frame(
                semantic_context, acoustic_context, temperature=0.0, generator=generator
            )
        generated.append(frame[0])
        history = torch.cat((history, frame[0, :, None]), dim=1)
    response_acoustic = torch.stack(generated, dim=1)
    del acoustic_model, package, history
    torch.cuda.empty_cache()

    decoder = AssistantDecoder()
    package = torch.load(args.decoder_checkpoint, map_location="cpu", weights_only=True)
    decoder.load_trainable_state(package["state"]).eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        response_wave = (
            decoder(response_semantic[None], response_acoustic[None])[0, 0].float().cpu().numpy()
        )
    expected_samples = len(response) * frame_samples
    response_wave = response_wave[:expected_samples]
    sf.write(args.output / "response.wav", response_wave, 24000)
    pause = np.zeros(int(0.4 * 24000), dtype=np.float32)
    conversation = np.concatenate((prompt_wave, pause, response_wave))
    sf.write(args.output / "conversation.wav", conversation, 24000)
    np.savez_compressed(
        args.output / "response-tokens.npz",
        semantic=response_semantic.cpu().numpy().astype("<u2"),
        acoustic=response_acoustic.cpu().numpy().astype("<u2"),
    )
    report = {
        "prompt_text": args.prompt,
        "prompt_seconds": len(prompt_wave) / 24000,
        "prompt_semantic_frames": int(len(prompt_semantic)),
        "response_seconds": len(response_wave) / 24000,
        "response_semantic_frames": len(response),
        "semantic_unique": int(response_semantic.unique().numel()),
        "acoustic_unique_by_codebook": [
            int(response_acoustic[index].unique().numel()) for index in range(7)
        ],
        "generated_end_turn": ended,
        "semantic_temperature": args.temperature,
        "semantic_top_k": args.top_k,
        "acoustic_decoding": "greedy",
        "acoustic_checkpoint": str(acoustic_checkpoint),
        "semantic_checkpoint": str(args.semantic_checkpoint),
        "decoder_checkpoint": str(args.decoder_checkpoint),
        "files": [
            "prompt.wav",
            "response.wav",
            "conversation.wav",
            "response-tokens.npz",
        ],
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
