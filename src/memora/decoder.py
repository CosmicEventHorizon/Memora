import torch
from moshi.models.loaders import get_mimi
from torch import nn

from .codec import DEFAULT_WEIGHTS


class AssistantDecoder(nn.Module):
    def __init__(self, device="cuda", weights=DEFAULT_WEIGHTS):
        super().__init__()
        self.mimi = get_mimi(weights, device=device, num_codebooks=8)
        for p in self.mimi.parameters():
            p.requires_grad = False
        for module in (self.mimi.decoder_transformer, self.mimi.decoder):
            for p in module.parameters():
                p.requires_grad = True
        self.semantic = nn.Embedding(2048, 512, device=device)
        nn.init.normal_(self.semantic.weight, std=0.02)
        self.semantic_scale = nn.Parameter(torch.tensor(0.0, device=device))

    def forward(self, semantic, acoustic):
        codes = torch.cat((semantic[:, None], acoustic), 1)
        emb = self.mimi.decode_latent(codes)
        emb = emb + torch.tanh(self.semantic_scale) * self.semantic(semantic).transpose(1, 2)
        emb = self.mimi._to_encoder_framerate(emb)
        if self.mimi.decoder_transformer is not None:
            (emb,) = self.mimi.decoder_transformer(emb)
        return self.mimi.decoder(emb)

    def trainable_state(self):
        return {
            "semantic": self.semantic.state_dict(),
            "semantic_scale": self.semantic_scale.detach().cpu(),
            "decoder_transformer": self.mimi.decoder_transformer.state_dict(),
            "decoder": self.mimi.decoder.state_dict(),
        }

    def load_trainable_state(self, state):
        self.semantic.load_state_dict(state["semantic"])
        self.semantic_scale.data.copy_(state["semantic_scale"])
        self.mimi.decoder_transformer.load_state_dict(state["decoder_transformer"])
        self.mimi.decoder.load_state_dict(state["decoder"])
        return self
