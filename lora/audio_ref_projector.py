"""Learned reference-audio projection adapter (B2) for HunyuanVideo-Foley.

Maps a CLAP audio embedding (512-d, shared audio<->text space) into the DiT's text
conditioning hidden space (condition_dim, e.g. 768) as `k_tokens` extra cross-attention
tokens. Trained with the base model FROZEN (AP-Adapter-lite). At inference it replaces the
training-free zero-pad bridge in utils._append_reference_token with a calibrated,
in-distribution token, so the reference influence is semantically precise rather than a
raw perturbation.

Design notes
------------
- The final linear is ZERO-initialized, so an untrained adapter emits ~zero tokens: it
  starts as (almost) a no-op and training moves it off zero only as it reduces loss.
  (At init the appended zero token still becomes cond_in.bias inside the model — one extra
  constant token among ~77 — a negligible perturbation that training absorbs immediately.)
- `k_tokens > 1` gives the cross-attention more capacity to carry reference character; the
  base already attends to a padded text K/V sequence, so a few extra tokens are cheap.
- Feed the SAME reference construction used at inference (a multi-clip centroid, see
  hunyuanvideo_foley.utils.feature_utils.clap_centroid) so train/test stay consistent.
"""
import torch
import torch.nn as nn


class AudioRefProjector(nn.Module):
    def __init__(self, in_dim: int = 512, cond_dim: int = 768, hidden: int = 1024, k_tokens: int = 1):
        super().__init__()
        self.in_dim = in_dim
        self.cond_dim = cond_dim
        self.hidden = hidden
        self.k_tokens = k_tokens
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, cond_dim * k_tokens),
        )
        # Zero-init the output layer -> adapter starts as a near no-op.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @property
    def config(self) -> dict:
        return {"in_dim": self.in_dim, "cond_dim": self.cond_dim,
                "hidden": self.hidden, "k_tokens": self.k_tokens}

    def forward(self, ref_embed: torch.Tensor) -> torch.Tensor:
        """ref_embed: [B, in_dim] (L2-normalized CLAP audio centroid).

        Returns [B, k_tokens, cond_dim] tokens to append to the text conditioning sequence.
        """
        if ref_embed.dim() == 1:
            ref_embed = ref_embed.unsqueeze(0)
        x = self.net(ref_embed)                       # [B, k_tokens * cond_dim]
        return x.view(x.shape[0], self.k_tokens, self.cond_dim)


def save_projector(projector: AudioRefProjector, path: str, meta: dict | None = None):
    torch.save({"state_dict": projector.state_dict(),
                "config": projector.config,
                "meta": meta or {}}, path)


def load_projector(path: str, map_location="cpu"):
    """Returns (projector, meta). Reconstructs architecture from the saved config."""
    ckpt = torch.load(path, map_location=map_location)
    projector = AudioRefProjector(**ckpt["config"])
    projector.load_state_dict(ckpt["state_dict"])
    projector.eval()
    return projector, ckpt.get("meta", {})


def append_ref_tokens(text_rep: torch.Tensor, uncond_rep: torch.Tensor, tokens: torch.Tensor):
    """Append learned reference tokens to the (already padded) text K/V sequence.

    Mirrors utils._append_reference_token's contract but takes ready-made tokens from the
    projector instead of the crude zero-pad bridge. A neutral (zero) token block is appended
    to the unconditional branch so the two CFG halves keep matching length and guidance
    carries the reference signal. Use this at INFERENCE once a projector is trained.

    text_rep / uncond_rep: [B, T, D];  tokens: [B, k, D] (or [k, D] / [1, k, D] -> broadcast)
    """
    B, T, D = text_rep.shape
    tok = tokens.to(device=text_rep.device, dtype=text_rep.dtype)
    if tok.dim() == 2:
        tok = tok.unsqueeze(0)
    if tok.shape[0] == 1 and B > 1:
        tok = tok.expand(B, -1, -1)
    k = tok.shape[1]
    zero = torch.zeros(B, k, D, device=uncond_rep.device, dtype=uncond_rep.dtype)
    return torch.cat([text_rep, tok], dim=1), torch.cat([uncond_rep, zero], dim=1)
