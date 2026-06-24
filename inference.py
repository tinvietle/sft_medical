#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM, PeftConfig
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig

from sft import SYSTEM_PROMPT, get_torch_dtype, load_env


text = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained SFT LoRA/QLoRA adapter.",
    )
    parser.add_argument(
        "--adapter-dir",
        required=True,
        help="Directory containing the trained adapter.",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Input case text. If omitted, the module-level `text` variable is used.",
    )
    parser.add_argument(
        "--text-file",
        default=None,
        help="Optional file containing the input case text.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. Ignored when --do-sample is not set.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter. Ignored when --do-sample is not set.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling. Default generation is greedy decoding.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16",
        help="Compute dtype for model loading.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Attention implementation passed to transformers.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow remote model code when loading base model artifacts.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit loading and use standard precision.",
    )
    return parser.parse_args()


def resolve_input_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        value = args.text.strip()
    elif args.text_file is not None:
        value = Path(args.text_file).read_text(encoding="utf-8").strip()
    else:
        value = text.strip()

    if not value:
        raise ValueError("No input text provided. Set `text = \"\"\"...\"\"\"`, or pass --text/--text-file.")
    return value


def build_quantization_config(
    base_model_name_or_path: str,
    *,
    dtype_name: str,
    trust_remote_code: bool,
    no_4bit: bool,
) -> BitsAndBytesConfig | None:
    config = AutoConfig.from_pretrained(
        base_model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    has_builtin_quantization = getattr(config, "quantization_config", None) is not None
    if has_builtin_quantization:
        print("Quantization mode: model-provided", flush=True)
        return None
    if no_4bit:
        print("Quantization mode: disabled", flush=True)
        return None

    print("Quantization mode: bitsandbytes-4bit", flush=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=get_torch_dtype(dtype_name),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def load_tokenizer(adapter_dir: str, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def main() -> None:
    load_env()
    args = parse_args()
    input_text = resolve_input_text(args)
    adapter_dir = str(Path(args.adapter_dir).resolve())

    peft_config = PeftConfig.from_pretrained(adapter_dir)
    quantization_config = build_quantization_config(
        peft_config.base_model_name_or_path,
        dtype_name=args.dtype,
        trust_remote_code=args.trust_remote_code,
        no_4bit=args.no_4bit,
    )
    torch_dtype = get_torch_dtype(args.dtype)

    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_dir,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model.eval()

    tokenizer = load_tokenizer(adapter_dir, trust_remote_code=args.trust_remote_code)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_text},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    with torch.no_grad():
        output_ids = model.generate(**model_inputs, **generation_kwargs)

    prompt_length = model_inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][prompt_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    print(response)


if __name__ == "__main__":
    main()
