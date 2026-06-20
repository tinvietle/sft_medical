# SFT Training

This repo trains an SFT LoRA/QLoRA adapter from local JSON files in `smol_json`.

## Dataset Format

Each JSON file must contain:

- `full_prompt`
- `reasoning`
- `answer`

The system prompt is defined in `sft.py`.

## Installation

```bash
pip install -r requirements.txt
```

## Required CLI Arguments

The training script requires these flags:

- `--model-id`
- `--output-dir`
- `--wandb-project`

`--dataset-dir` defaults to `smol_json`.

## Example

```bash
python sft.py \
  --dataset-dir smol_json \
  --model-id Qwen/Qwen3-0.6B \
  --output-dir Qwen3-0.6B-SFT \
  --wandb-project MedCaseReasoning
```

## Environment Variables

Use `.env.example` as a template for:

- `HF_TOKEN`
- `WANDB_API_KEY` or `WANDB_TOKEN`

## Notes

- Files with both empty `reasoning` and empty `answer` are skipped.
- Use `--disable-wandb` if you do not want Weights & Biases logging.
- Use `--no-push-to-hub` if you do not want to upload to Hugging Face Hub.

