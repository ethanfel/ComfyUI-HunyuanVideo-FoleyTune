# Timed Audio Cues for HunyuanVideo-Foley — Research & Design Reference

**Date:** 2026-04-11
**Status:** Research complete, not yet implemented
**Prerequisite:** LoRA training pipeline (complete), first sweep validation (in progress)

## Goal

Add temporal text conditioning to Foley audio generation — e.g., `"at 0:01.54 she screams with pleasure"` — so the model places specific audio events at precise timestamps.

## Current Architecture: No Temporal Text Control

### Three Conditioning Streams

| Stream | Model | Shape (8s clip) | Resolution | Temporal? |
|--------|-------|-----------------|------------|-----------|
| Visual | SigLIP2 | `[B, 64, 768]` | 8fps | Implicit via RoPE position |
| Sync | Synchformer | `[B, 192, 768]` | 25fps | Motion-based sync only |
| Text | CLAP | `[B, ≤77, 768]` | Global | **No** — all tokens attend to all audio frames |

### Text Pathway Detail

```
Text prompt
  → CLAP tokenizer (max 77 tokens)
  → ClapTextModelWithProjection → [B, N_text, 768]
  → ConditionProjection (768 → 1536, SiLU, 1536 → 1536)
  → Cross-attention in 18 TwoStreamCABlocks:
      Query:  audio_cross_q (audio stream) + v_cond_cross_q (visual stream)
      Key/Value: text_cross_kv (shared text embeddings)
      Heads: 12, head_dim: 128
```

Every text token attends to every audio frame equally. No mechanism exists to constrain
"this text applies at this time."

### Cross-Attention Architecture (TwoStreamCABlock)

```
Layers per block (18 blocks total):
  audio_self_attn_qkv    [1536 → 4608]   Audio self-attention Q/K/V
  audio_self_proj        [1536 → 1536]   Audio self-attention output
  audio_cross_q          [1536 → 1536]   Audio queries for text cross-attn
  v_cond_attn_qkv        [1536 → 4608]   Visual self-attention Q/K/V
  v_cond_self_proj       [1536 → 1536]   Visual self-attention output
  v_cond_cross_q         [1536 → 1536]   Visual queries for text cross-attn
  text_cross_kv          [1536 → 3072]   Text key/value (shared by audio+visual)
  audio_mlp.fc1/fc2      [1536 ↔ 6144]   Audio feed-forward
  v_cond_mlp.fc1/fc2     [1536 ↔ 6144]   Visual feed-forward
```

### Tensor Shapes Reference (8s generation, XXL model)

| Component | Shape | Notes |
|-----------|-------|-------|
| Audio latents (raw) | `[B, 128, 750]` | DAC hop=512, 48kHz |
| Audio after PatchEmbed1D | `[B, 375, 1536]` | Embedded into hidden_size |
| SigLIP2 visual (projected) | `[B, 64, 1536]` | After visual_proj (SwiGLU) |
| Synchformer sync | `[B, 192, 768]` | Grouped in 8-frame chunks |
| CLAP text (projected) | `[B, ≤77, 1536]` | After ConditionProjection |
| Cross-attn Q (concat) | `[B, 375+64, 12, 128]` | Audio+Visual queries |
| Cross-attn K/V | `[B, ≤77, 12, 128]` | Text keys/values |

### Modulation (DiT)

Both audio and visual streams receive 9-factor DiT modulation from diffusion timestep `vec`.
Sync features (when `add_sync_feat_to_audio=True`) are added to `vec` at layer 0:
`sync_vec + vec.unsqueeze(1)` — provides temporal amplitude modulation but no semantic content.

---

## Approach 1: Timestamp Tokens in Text (Recommended First Step)

### Concept

Embed timestamps directly in the text prompt as special tokens:
```
"<T00> ambient breathing <T01> she screams <T03> wet slurping <T06> moaning"
```

### Implementation

**Modify `encode_text_feat()` in `feature_utils.py` (line 134):**

Preprocess the prompt to inject timestamp tokens before CLAP tokenization. CLAP's
tokenizer will treat them as unknown tokens, but the LoRA-trained cross-attention
layers can learn to associate these token positions with temporal positions in the
audio stream via RoPE.

**No architecture changes required.** The existing cross-attention in TwoStreamCABlock
handles variable-length text sequences. The LoRA targets that matter:

```python
FOLEY_TARGET_PRESETS["temporal_text"] = (
    "text_cross_kv",        # Text learns to encode temporal position
    "audio_cross_q",        # Audio learns to query by time
    "audio_cross_proj",     # Audio cross-attention output
    "audio_mlp.fc1",        # Post-attention refinement
    "audio_mlp.fc2",
)
```

### Training Data Format

```json
{
  "prompt": "<T00> ambient breathing <T01> she screams <T03> wet slurping",
  "events": [
    {"time": 0.0, "label": "ambient breathing"},
    {"time": 1.54, "label": "she screams"},
    {"time": 3.2, "label": "wet slurping"}
  ]
}
```

The `events` field is for metadata/future use. Training uses the formatted `prompt` string.

### Expected Results

- **Coarse temporal control:** 1-2 second granularity
- **Why limited:** CLAP wasn't pretrained on timestamp tokens — they're out-of-distribution.
  The model can learn rough associations through LoRA fine-tuning but won't achieve
  frame-level precision through text alone.
- **Still valuable:** Even coarse control ("scream in first half" vs "scream at end")
  would be a significant capability improvement.

### Effort

- Text preprocessing: ~1 day
- Dataset with timestamps: ~2-3 days (manual annotation or onset detection)
- LoRA training: uses existing pipeline
- **Total: ~1 week**

---

## Approach 2: Parallel Temporal Event Stream (Most Robust)

### Concept

Add a dedicated cross-attention pathway for timed events, parallel to the existing
text pathway. Each event has its own embedding + explicit temporal position.

### Architecture

```
Input events: [{time: 1.54, text: "she screams"}, {time: 3.2, text: "footsteps"}]

Processing:
1. Encode each event text via CLAP → [N_events, 768]
2. Add learned temporal position embedding (quantized to audio frame rate)
3. Project to hidden_size → [N_events, 1536]
4. New cross-attention in TwoStreamCABlock:
     Query:  audio stream [B, 375, 1536]
     Key/Value: temporal events [B, N_events, 1536]
     Position: RoPE with event timestamps mapped to audio frame positions
```

### New Layers (per TwoStreamCABlock, 18 blocks)

```python
# Add to TwoStreamCABlock.__init__:
self.temporal_event_q = nn.Linear(1536, 1536)      # Audio queries events
self.temporal_event_kv = nn.Linear(1536, 3072)      # Event key/value
self.temporal_event_proj = nn.Linear(1536, 1536)     # Output projection
```

**New LoRA targets: 3 layers × 18 blocks = 54 new Linear layers**

### Temporal Position Encoding

Events map to audio frame positions:
```
event_time = 1.54s
audio_frame_rate = 375 frames / 8s = 46.875 fps
event_frame = int(1.54 * 46.875) = 72
```

Use the same RoPE mechanism as visual features, but with event-specific frequencies
so the model learns "event at frame 72" → "audio content at frame 72."

### Key Modification Points

| File | Location | Change |
|------|----------|--------|
| `hifi_foley.py:707` | `forward()` signature | Add `temporal_events` parameter |
| `hifi_foley.py:271-319` | `TwoStreamCABlock` | Add temporal event cross-attention |
| `feature_utils.py` | New function | `encode_temporal_events()` |
| `lora/lora.py:18-56` | Target presets | Add `temporal_event` preset |
| `nodes_lora.py` | Feature extractor | Accept event annotations |
| `lora/train.py` | `flow_matching_loss()` | Pass events through model forward |

### Expected Results

- **Fine temporal control:** ~40ms resolution (matching 25Hz sync rate)
- **Per-event conditioning:** Each event independently positioned
- **Composable:** Multiple events can overlap or be spread across the 8s window

### Effort

- Architecture changes: ~1 week
- Feature extractor + dataset format: ~3 days
- Training pipeline updates: ~2 days
- Training + iteration: ~1-2 weeks
- **Total: 3-4 weeks**

---

## Approach 3: Repurpose Sync Features (Not Recommended)

Synchformer is frozen, purely visual, and designed for motion sync — not semantic events.
Retraining it for text events would be high effort for unclear benefit and would lose
the motion synchronization capability.

---

## Decision Matrix

| Criteria | Approach 1 (Tokens) | Approach 2 (Event Stream) |
|----------|---------------------|---------------------------|
| Architecture changes | None | New cross-attn layers |
| Training data | Timestamped prompts | Event annotations |
| Temporal precision | ~1-2s | ~40ms |
| Implementation effort | 1 week | 3-4 weeks |
| Risk | Low (fallback: ignore timestamps) | Medium (new untested pathway) |
| LoRA compatible | Yes (existing targets) | Yes (new targets needed) |
| Inference overhead | None | +54 Linear layers per forward |

### Recommended Path

1. **Start with Approach 1** — validate that temporal text conditioning is learnable
2. If results show even coarse control, proceed to **Approach 2** for precision
3. Approach 1 training data (timestamped prompts) can be reused for Approach 2
