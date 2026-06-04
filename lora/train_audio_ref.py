"""Train the reference-audio projection adapter (B2) — SCAFFOLD.

Trains an AudioRefProjector (lora/audio_ref_projector.py) on top of a FROZEN
HunyuanVideo-Foley model so a reference-audio CLAP embedding becomes a calibrated
conditioning token (vs. the training-free zero-pad bridge in utils._append_reference_token).

Multi-clip + anti-copy-paste, by design
---------------------------------------
- Each training reference is a CENTROID of `k_clips` clips of the same performer
  (clap_centroid), matching how references are built at inference (FoleyTuneReferenceAudio).
- Two-stage curriculum (AC-Foley): Stage I may include the target's own clip (warmup, the
  model learns to *use* the token); Stage II samples only *different* clips of the same
  performer (forces clip-invariant performer character, prevents trivial copy-paste).

Integration boundary (this is a scaffold)
-----------------------------------------
- `model` + `deps` are passed in already loaded — obtain them from the ComfyUI loaders
  (FoleyTuneModelLoader / FoleyTuneDependenciesLoader) in a thin wrapper node, or load them
  in a standalone script. A `FoleyTuneAudioRefTrain` node would mirror FoleyTuneLoRATrain.
- TODO(you): provide `performer_of` to group clips by performer. Defaults to None-grouping
  (every reference falls back to self -> Stage I behaviour only). See the example helpers.
- Batch>1 collation is intentionally avoided: we accumulate gradient over single-sample
  forwards (correct, simple). Scale `grad_accum` for an effective batch.

Inference wiring (after training)
---------------------------------
Load with audio_ref_projector.load_projector(), compute tokens = projector(ref_embed), then
append via audio_ref_projector.append_ref_tokens() in place of the crude bridge inside
utils._append_reference_token (thread a projector handle through the sampler the same way
reference_embed is threaded today).
"""
import os
import re
import random

import torch
from loguru import logger

from .train import prepare_dataset, flow_matching_loss, sample_timesteps
from .audio_ref_projector import AudioRefProjector, save_projector
from hunyuanvideo_foley.utils.feature_utils import encode_audio_clap, clap_centroid
from hunyuanvideo_foley.utils.config_utils import AttributeDict

_AUDIO_EXTS = (".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".m4a")
_PRECISION = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


# --- performer grouping (pluggable — customize for your dataset) --------------

def performer_from_map(name_to_performer: dict):
    """Closure: look up a performer id from an explicit {clip_name: performer} mapping."""
    return lambda sample: name_to_performer.get(sample["name"])


def performer_from_prompt(tag_regex: str = r"\b([a-z]{2,4})\b"):
    """Closure: extract a performer tag from the prompt via regex (first match).

    Example for the `abd` performer convention; adjust the pattern to your tagging scheme.
    """
    pat = re.compile(tag_regex)
    def _of(sample):
        m = pat.search(sample.get("prompt", ""))
        return m.group(1) if m else None
    return _of


# --- reference construction ---------------------------------------------------

def _find_audio(data_dir, stem):
    for ext in _AUDIO_EXTS:
        p = os.path.join(data_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


@torch.no_grad()
def precompute_ref_embeds(dataset, deps, data_dir, device):
    """Encode each clip's RAW audio -> normalized CLAP embedding {name: [1,512]}.

    Uses raw audio (present alongside the .npz caches) rather than decoding DAC latents, so
    embeddings match what FoleyTuneReferenceAudio produces at inference.
    """
    import torchaudio
    model_dict = AttributeDict(dict(deps))
    model_dict["device"] = device
    deps["clap_audio_model"].to(device)
    embeds = {}
    try:
        for s in dataset:
            ap = _find_audio(data_dir, s["name"])
            if ap is None:
                logger.warning(f"[audioref] no audio for {s['name']}, skipping reference")
                continue
            wav, sr = torchaudio.load(ap)          # [C, N]
            wav = wav.unsqueeze(0)                  # [1, C, N]
            if sr != 48000:
                wav = torchaudio.functional.resample(wav, sr, 48000)
            embeds[s["name"]] = encode_audio_clap(wav, model_dict).cpu()  # [1, 512]
    finally:
        deps["clap_audio_model"].to("cpu")
    logger.info(f"[audioref] precomputed {len(embeds)} reference CLAP embeddings")
    return embeds


def build_groups(dataset, performer_of):
    groups, name_to_perf = {}, {}
    for s in dataset:
        perf = performer_of(s) if performer_of else None
        name_to_perf[s["name"]] = perf
        if perf is not None:
            groups.setdefault(perf, []).append(s["name"])
    n_perf = len(groups)
    logger.info(f"[audioref] grouped into {n_perf} performer(s); "
                f"{sum(1 for v in name_to_perf.values() if v is None)} ungrouped")
    return groups, name_to_perf


def sample_reference_embed(idx, dataset, groups, name_to_perf, embeds, k_clips, stage, rng):
    """Centroid reference embedding [1,512] for the target at `idx`.

    Stage 'II' excludes the target clip (different-clip -> anti copy-paste). Falls back to
    self-reference when the performer is unknown or has no other clips.
    """
    name = dataset[idx]["name"]
    perf = name_to_perf.get(name)
    pool = list(groups.get(perf, [])) if perf is not None else []
    if stage == "II":
        pool = [n for n in pool if n != name]
    pool = [n for n in pool if n in embeds]
    if not pool:
        pool = [name] if name in embeds else list(embeds.keys())
    chosen = rng.sample(pool, min(k_clips, len(pool)))
    return clap_centroid([embeds[n] for n in chosen])  # [1, 512]


# --- training -----------------------------------------------------------------

def train_audio_ref_projector(
    model, deps, cond_dim, data_dir, out_dir,
    steps=3000, lr=1e-4, grad_accum=4,
    k_tokens=1, hidden=1024,
    k_clips=3, stage1_frac=0.3,
    timestep_mode="logit_normal", precision="bf16",
    performer_of=None, save_every=500, seed=0,
    loss_kwargs=None,
):
    """Train the projector with the base model frozen. Returns the trained projector.

    cond_dim: the DiT text-conditioning dim (model.condition_dim, e.g. 768).
    """
    device = next(model.parameters()).device
    dtype = _PRECISION.get(precision, torch.bfloat16)
    cond_dim = int(cond_dim)
    os.makedirs(out_dir, exist_ok=True)
    loss_kwargs = loss_kwargs or {}

    # Data + references
    dataset = prepare_dataset(data_dir, deps["dac_model"], device, dtype=dtype)
    embeds = precompute_ref_embeds(dataset, deps, data_dir, device)
    groups, name_to_perf = build_groups(dataset, performer_of)

    # Freeze base; train projector only
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    projector = AudioRefProjector(in_dim=512, cond_dim=cond_dim, hidden=hidden, k_tokens=k_tokens).to(device)
    projector.train()
    opt = torch.optim.AdamW(projector.parameters(), lr=lr, weight_decay=0.0)

    rng = random.Random(seed)
    stage_switch = int(steps * stage1_frac)
    logger.info(f"[audioref] training {steps} steps; stage I<{stage_switch}, II after; "
                f"k_tokens={k_tokens} k_clips={k_clips} cond_dim={cond_dim}")

    for step in range(steps):
        stage = "I" if step < stage_switch else "II"
        opt.zero_grad(set_to_none=True)
        running = 0.0
        for _ in range(grad_accum):
            idx = rng.randrange(len(dataset))
            s = dataset[idx]
            x1 = s["latents"].to(device)
            clip_feat = s["clip_features"]
            sync_feat = s["sync_features"]
            text_feat = s["text_embedding"].to(device)

            ref = sample_reference_embed(idx, dataset, groups, name_to_perf, embeds,
                                         k_clips, stage, rng).to(device).float()
            tokens = projector(ref)                                  # [1, k, cond_dim]
            text_aug = torch.cat([text_feat, tokens.to(text_feat.dtype)], dim=1)

            t = sample_timesteps(1, timestep_mode, device, torch.float32)
            loss = flow_matching_loss(model, x1, t, clip_feat, sync_feat, text_aug,
                                      device, dtype, **loss_kwargs) / grad_accum
            loss.backward()
            running += loss.item()
        opt.step()

        if step % 50 == 0:
            logger.info(f"[audioref] step {step}/{steps} stage={stage} loss={running:.4f}")
        if save_every and step > 0 and step % save_every == 0:
            save_projector(projector, os.path.join(out_dir, f"audioref_step{step}.pt"),
                           meta={"step": step, "k_tokens": k_tokens, "k_clips": k_clips})

    final_path = os.path.join(out_dir, "audioref_final.pt")
    save_projector(projector, final_path,
                   meta={"steps": steps, "k_tokens": k_tokens, "k_clips": k_clips,
                         "stage1_frac": stage1_frac, "timestep_mode": timestep_mode})
    logger.info(f"[audioref] saved final projector -> {final_path}")
    return projector
