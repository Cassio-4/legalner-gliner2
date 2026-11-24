"""Merge a PEFT LoRA adapter into its GLiNER2 base model and save the full model.

Usage:
    python merge_lora_to_model.py output/EXP-03/<run>/best

The input folder must contain ``adapter_config.json`` and
``adapter_model.safetensors``. The base model is read from
``base_model_name_or_path`` in ``adapter_config.json``. The merged model is
written to ``<path>/best_merged`` and loads with ``AutoExtractor.from_pretrained``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from gliner2 import AutoExtractor
from peft import PeftModel


def merge(adapter_dir: Path, device: str = "cpu") -> Path:
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    for p in (config_path, weights_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing {p}")

    base_model_id = json.loads(config_path.read_text())["base_model_name_or_path"]
    out_dir = adapter_dir / "best_merged"

    print(f"Loading base model {base_model_id} (device={device})...")
    base = AutoExtractor.from_pretrained(base_model_id, map_location=device)

    print(f"Loading LoRA adapter from {adapter_dir}...")
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))

    print("Merging adapter into base weights...")
    merged = peft_model.merge_and_unload()
    merged.eval()

    print(f"Saving merged model to {out_dir}...")
    merged.save_pretrained(str(out_dir))
    print("Done.")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", type=Path, help="Folder with adapter_config.json and adapter_model.safetensors")
    parser.add_argument(
        "--device",
        default="cpu",
        help="map_location for loading the base model (default: cpu)",
    )
    args = parser.parse_args()

    with torch.no_grad():
        merge(args.path.resolve(), device=args.device)


if __name__ == "__main__":
    main()
