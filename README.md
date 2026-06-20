# SFT Training

This repo trains an SFT LoRA/QLoRA adapter from local JSON files in `smol_json`.

The Python environment is pinned for a CUDA 12.4-compatible PyTorch stack, which matches NVIDIA driver branch 550 on the workstation.

## Dataset Format

Each JSON file must contain:

- `full_prompt`
- `reasoning`
- `answer`

The system prompt is defined in `sft.py`.

## Installation

```bash
uv python install 3.12
uv venv --python 3.12
uv pip install -r requirements.txt
uv pip install --index-url https://download.pytorch.org/whl/cu124 -r requirements-cu124.txt
```

Optional:

```bash
uv pip install wandb
```

The pinned CUDA 12.4 PyTorch install is separated into `requirements-cu124.txt`:

- `torch==2.6.0+cu124`
- `torchvision==0.21.0+cu124`
- `torchaudio==2.6.0+cu124`

through the PyTorch CUDA 12.4 wheel index.

## Required CLI Arguments

The training script requires these flags:

- `--model-id`
- `--output-dir`
- `--wandb-project`

`--dataset-dir` defaults to `smol_json`.

## Example

```bash
.venv/bin/python sft.py \
  --dataset-dir smol_json \
  --model-id Qwen/Qwen3-0.6B \
  --output-dir Qwen3-0.6B-SFT \
  --wandb-project MedCaseReasoning
```

## Environment Variables

Use `.env.example` as a template for:

- `HF_TOKEN`
- `WANDB_API_KEY` or `WANDB_TOKEN`

The script automatically loads `.env` at startup.

## Notes

- Files with both empty `reasoning` and empty `answer` are skipped.
- Use `--disable-wandb` if you do not want Weights & Biases logging.
- Use `--no-push-to-hub` if you do not want to upload to Hugging Face Hub.
- `wandb` is not required unless you want W&B logging.
- `.python-version` pins the project to Python 3.12 for `uv`.
