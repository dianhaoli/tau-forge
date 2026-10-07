"""Merge a LoRA checkpoint into its base model (bf16) and save model + tokenizer,
ready for `vllm serve`.

    uv run --extra train python scripts/merge_lora.py \\
        --adapter /workspace/runs/ep-main/checkpoint-75 --out /workspace/runs/ep-main/merged-75
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

p = argparse.ArgumentParser()
p.add_argument("--adapter", required=True)
p.add_argument("--out", required=True)
p.add_argument("--base", default=None, help="Default: base_model_name_or_path from adapter_config.json.")
args = p.parse_args()

base_id = args.base or json.loads((Path(args.adapter) / "adapter_config.json").read_text())["base_model_name_or_path"]
base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(base, args.adapter)
merged = model.merge_and_unload()
merged.save_pretrained(args.out, safe_serialization=True)
AutoTokenizer.from_pretrained(base_id).save_pretrained(args.out)
GenerationConfig.from_pretrained(base_id).save_pretrained(args.out)
print(f"merged {args.adapter} into {base_id} -> {args.out}")
