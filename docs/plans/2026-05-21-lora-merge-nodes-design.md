# FoleyTune LoRA Merge Nodes Design

**Date:** 2026-05-21
**Goal:** Add two LoRA merge nodes to FoleyTune — a manual-strategy Merger and an auto-tuning optimizer — that operate natively on `FOLEYTUNE_MODEL` without depending on the ZImage LoRA Merger.

## Architecture

### Files

- **`lora/merge_math.py`** (~300 lines) — self-contained merge math extracted from `lora_optimizer.py`. No external dependencies beyond PyTorch.
- **`nodes_merge.py`** — two ComfyUI nodes: `FoleyTuneLoRAMerger` and `FoleyTuneLoRAAutoTuner`.
- **`__init__.py`** — updated to register the new nodes.

### Merge Math Module (`lora/merge_math.py`)

Standalone functions, no class state:

- `compute_delta(state_dict, rank, alpha, strength, use_rslora) -> dict[str, Tensor]` — for each `lora_A`/`lora_B` pair, compute `B @ A * scaling * strength`. Returns `{layer_name: delta_tensor}`.
- `analyze_conflict(deltas: list[Tensor]) -> dict` — pairwise cosine similarity, sign conflict ratio, excess conflict (conflict minus 0.5 baseline for orthogonal pairs).
- `merge_weighted_average(deltas: list[Tensor], weights: list[float]) -> Tensor`
- `merge_ties(deltas: list[Tensor], weights: list[float], conflict_threshold: float) -> Tensor` — trim, elect sign, disjoint merge.
- `merge_slerp(delta_a: Tensor, delta_b: Tensor, t: float) -> Tensor` — spherical interpolation (2-LoRA only).
- `classify_relationship(cos_sim, conflict_ratio, excess_conflict) -> str` — returns `"consensus"`, `"orthogonal"`, or `"conflicting"`.
- `auto_strength_normalize(deltas: list[Tensor], strengths: list[float]) -> list[float]` — energy-based normalization so combined LoRAs don't over/under-saturate.

Thresholds use the `dit` arch preset values from `lora_optimizer.py` (Foley is a DiT variant):
- `consensus_cos_sim_min = 0.5`, `consensus_conflict_max = 0.15`
- `orthogonal_cos_sim_max = 0.25`, `orthogonal_conflict_max = 0.60`
- `ties_conflict_threshold = 0.25`

### Prefix Grouping

FoleyTune LoRA keys follow the pattern:
```
triple_blocks.N.<suffix>.base.lora_A
triple_blocks.N.<suffix>.base.lora_B
```

Where `<suffix>` is one of `audio_self_attn_qkv`, `audio_self_proj`, `audio_cross_q`, `audio_cross_proj`, `text_cross_kv`, `v_cond_attn_qkv`, `v_cond_self_proj`, `v_cond_cross_q`, `v_cond_cross_proj`, `audio_mlp.fc1`, `audio_mlp.fc2`, `v_cond_mlp.fc1`, `v_cond_mlp.fc2`.

**Grouping strategy:** group by `triple_blocks.N` (block index 0–11). Each block becomes one prefix group with up to 14 target layers. The merge strategy is decided per block — same approach as ZImage's per-prefix optimization.

### Node 1: `FoleyTuneLoRAMerger`

Manual strategy selection. User picks the merge method.

**Inputs:**
| Name | Type | Notes |
|------|------|-------|
| `hunyuan_model` | `FOLEYTUNE_MODEL` | Required |
| `lora_name_1` | loras dropdown | Required |
| `strength_1` | FLOAT 0.0–2.0 | Required, default 1.0 |
| `lora_name_2` | loras dropdown | Required |
| `strength_2` | FLOAT 0.0–2.0 | Required, default 1.0 |
| `lora_name_3` | loras dropdown | Optional |
| `strength_3` | FLOAT 0.0–2.0 | Optional, default 1.0 |
| `lora_name_4` | loras dropdown | Optional |
| `strength_4` | FLOAT 0.0–2.0 | Optional, default 1.0 |
| `merge_strategy` | enum | `weighted_average`, `ties`, `slerp` (default: `ties`) |

**Output:** `(FOLEYTUNE_MODEL, STRING)` — merged model + combined prompts from LoRA metadata.

**Flow:**
1. Load each LoRA via `_load_adapter_checkpoint`, infer rank/alpha/target from metadata.
2. Compute deltas per target layer using `compute_delta`.
3. Group by block prefix (`triple_blocks.N`).
4. Apply the selected strategy to every group.
5. Add merged deltas to a `deepcopy` of the base model's weights.
6. Return model + concatenated prompts.

SLERP is restricted to exactly 2 LoRAs (the node falls back to TIES if >2 and SLERP is selected).

### Node 2: `FoleyTuneLoRAAutoTuner`

Automatic per-block strategy selection with conflict analysis.

**Inputs:**
| Name | Type | Notes |
|------|------|-------|
| `hunyuan_model` | `FOLEYTUNE_MODEL` | Required |
| `lora_name_1..4` | loras dropdown | 1–2 required, 3–4 optional |
| `strength_1..4` | FLOAT | Same as Merger |
| `auto_strength` | enum | `disabled`, `enabled` (default: `disabled`) |
| `sparsification` | enum | `disabled`, `dare`, `conflict_aware` (default: `disabled`) |
| `sparsification_density` | FLOAT 0.1–1.0 | Default 0.7 |

**Output:** `(FOLEYTUNE_MODEL, STRING, STRING)` — merged model + prompts + analysis report.

**Flow:**
1. Load LoRAs, compute deltas (same as Merger).
2. For each block prefix, run `analyze_conflict` on the stacked deltas.
3. Use `classify_relationship` to auto-select strategy per block:
   - `consensus` → `weighted_average`
   - `orthogonal` → `slerp` (2 LoRAs) or `weighted_average` (3+)
   - `conflicting` → `ties`
4. If `auto_strength=enabled`, run `auto_strength_normalize` before merging.
5. If sparsification enabled, apply DARE or conflict-aware mask before merge.
6. Apply merged deltas to model.
7. Build a human-readable report (per-block strategy + metrics).

### Integration

Both nodes use `merge_lora_into_weights`-style application: compute the full merged delta tensor per layer, then `weight.data += delta.to(weight.dtype)`. Single `deepcopy` of the model.

The existing `FoleyTuneLoRAStacker` remains unchanged — it's the simple sequential chain option.

### Registration

Add to `__init__.py` NODE_CLASS_MAPPINGS:
```python
"FoleyTuneLoRAMerger": FoleyTuneLoRAMerger,
"FoleyTuneLoRAAutoTuner": FoleyTuneLoRAAutoTuner,
```

Display names: `"FoleyTune LoRA Merger"`, `"FoleyTune LoRA AutoTuner"`.

## Out of Scope

- Community cache / HF upload. This is a local-only merge.
- Phase 2 candidate search (scoring multiple merge configurations). The AutoTuner uses a single-pass heuristic, not the full Phase 1 + Phase 2 pipeline from ZImage.
- CLIP handling — Foley has no CLIP component.
- LoKr/LoHa support — FoleyTune only trains standard LoRA (A/B matrices).
- SingleStreamBlock LoRA targeting — not currently used by any FoleyTune preset.
