#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from sft import SYSTEM_PROMPT, get_torch_dtype, load_env


text = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference by loading a base model first, then attaching a LoRA/QLoRA adapter from Hugging Face.",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Base model repo ID or local path.",
    )
    parser.add_argument(
        "--adapter-id",
        required=True,
        help="Hugging Face repo ID for the trained adapter.",
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
        "--live-chat",
        action="store_true",
        help="Run an interactive stateless chat loop. Type 'q' to quit.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional maximum number of new tokens to generate. Omit to leave generation uncapped.",
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


def get_hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def resolve_input_text(args: argparse.Namespace) -> str:
    if args.live_chat:
        return ""

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
    model_id: str,
    *,
    dtype_name: str,
    trust_remote_code: bool,
    no_4bit: bool,
    token: str | None,
) -> BitsAndBytesConfig | None:
    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        token=token,
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


def load_tokenizer(
    model_id: str,
    *,
    trust_remote_code: bool,
    token: str | None,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def generate_response(
    fine_tuned_model,
    tokenizer,
    *,
    query: str,
    max_new_tokens: int | None,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer(prompt, return_tensors="pt").to(fine_tuned_model.device)

    generation_kwargs = {
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if max_new_tokens is not None:
        generation_kwargs["max_new_tokens"] = max_new_tokens
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.no_grad():
        output_ids = fine_tuned_model.generate(**model_inputs, **generation_kwargs)

    prompt_length = model_inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][prompt_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main() -> None:
    load_env()
    args = parse_args()
    input_text = resolve_input_text(args)
    hf_token = get_hf_token()

    quantization_config = build_quantization_config(
        args.model_id,
        dtype_name=args.dtype,
        trust_remote_code=args.trust_remote_code,
        no_4bit=args.no_4bit,
        token=hf_token,
    )
    torch_dtype = get_torch_dtype(args.dtype)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        quantization_config=quantization_config,
        device_map="auto",
        token=hf_token,
    )
    fine_tuned_model = PeftModel.from_pretrained(
        base_model,
        args.adapter_id,
        token=hf_token,
    )
    fine_tuned_model.eval()

    tokenizer = load_tokenizer(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        token=hf_token,
    )

    if args.live_chat:
        while True:
            query = input("query> ").strip()
            if query == "q":
                break
            if not query:
                continue

            response = generate_response(
                fine_tuned_model,
                tokenizer,
                query=query,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(response)
        return

    response = generate_response(
        fine_tuned_model,
        tokenizer,
        query=input_text,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(response)


if __name__ == "__main__":
    main()
