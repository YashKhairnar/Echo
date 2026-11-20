# Experiment 1 datasets

This folder contains the generated datasets for the first end-to-end training
experiment:

- `intent_aware_dataset.jsonl`: transitions collected with optional neighbor
  intent requests.
- `random_no_message_dataset.jsonl`: random ego actions without messages.
- `combined_world_model_dataset.jsonl`: the merged file used to train MLPx.

The JSONL files are generated artifacts and are excluded from Git. Recreate
them with the collectors under `scripts/data_collection/`.
