import io
import json
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
from piper import PiperVoice
from piper.config import PiperConfig
from scipy.signal import resample_poly


def load_voice(path: Path, device: str = "cuda") -> PiperVoice:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        providers.insert(0, ("CUDAExecutionProvider", {"device_id": 0}))
    elif device != "cpu":
        raise ValueError(f"Unsupported device: {device}")
    session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    if device == "cuda" and session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("CUDAExecutionProvider is unavailable")
    config = PiperConfig.from_dict(json.loads(Path(f"{path}.json").read_text()))
    return PiperVoice(session=session, config=config)


def synthesize(voice: PiperVoice, text: str, sample_rate: int = 24_000) -> np.ndarray:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        voice.synthesize_wav(text, output)
    buffer.seek(0)
    with wave.open(buffer) as source:
        source_rate = source.getframerate()
        audio = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    waveform = audio.astype(np.float32) / 32_768
    if source_rate != sample_rate:
        waveform = resample_poly(waveform, sample_rate, source_rate).astype(np.float32)
    if len(waveform) < sample_rate // 10 or not np.isfinite(waveform).all():
        raise ValueError("TTS produced invalid audio")
    if np.sqrt(np.mean(waveform**2)) < 1e-4:
        raise ValueError("TTS produced silent audio")
    return waveform
