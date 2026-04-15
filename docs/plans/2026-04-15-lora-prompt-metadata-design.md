# LoRA Prompt Metadata Design

**Date:** 2026-04-15
**Status:** Approved

## Goal

Embed training prompts in the LoRA checkpoint so the Loader can output
them at inference time. Users shouldn't have to remember or look up what
prompts were used during training.

## Problem

Prompts are stored in .npz feature files and used during training as
CLAP embeddings, but they are not saved in the .pt checkpoint. After
training, the prompt information is lost unless the user manually
tracks it.

## Approach

Add a `"prompts"` key to the existing `meta` dict in the checkpoint.
Store unique prompts sorted by frequency (most used first). The LoRA
Loader reads it and outputs as a newline-joined string.

## Changes

### Trainer (`_train_inner` in nodes_lora.py)

After `prepare_dataset()`, collect unique prompts sorted by frequency:

```python
from collections import Counter
prompt_counts = Counter(d["prompt"] for d in dataset)
meta["prompts"] = [p for p, _ in prompt_counts.most_common()]
```

No changes to `save_checkpoint()` — the meta dict is already saved.
Also appears in `meta.json` for human readability.

### Loader (`load_adapter` in nodes_lora.py)

- Read `prompts = meta.get("prompts", [])` from checkpoint
- Add second output: `RETURN_TYPES = ("FOLEY_MODEL", "STRING")`
  with `RETURN_NAMES = ("model", "prompts")`
- Join with newlines: `"\n".join(prompts)`

### Backward Compatibility

- Old checkpoints without `"prompts"` in meta: Loader outputs empty string
- Old workflows not wiring the `prompts` output: unaffected (ComfyUI
  ignores unused outputs)

### Dependencies

None new.
