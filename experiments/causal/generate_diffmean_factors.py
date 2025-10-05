#!/usr/bin/env python3
import sys
import os
import json
import argparse
import random
import numpy as np
import torch
from datetime import datetime

from transformer_lens import HookedTransformer
from experiments.baselines.diffmean import DiffMean


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures deterministic behavior


def log(txt: str):
    print(f"[{datetime.now()}] {txt}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute DiffMean concept vectors and save per (layer, h_row). "
                    "All configurable parameters are provided via CLI."
    )
    # Core config (required)
    p.add_argument("--model", required=True,
                   help="HF repo id for HookedTransformer.from_pretrained (e.g., 'gemma-2-2b', 'meta-llama/Llama-3.1-8B').")
    p.add_argument("--concept-data", required=True,
                   help="Path to JSON with entries containing keys 'layer', 'h_row', "
                        "'activating_sentences', 'neutral_sentences'.")
    p.add_argument("--concept-dir", required=True,
                   help="Directory to save concept vectors .pt files.")
    p.add_argument("--mode", required=True, choices=["mlp_out", "mlp", "resid", "attn_out"],
                   help="Activation mode to use for DiffMean (your code uses 'mlp_out').")

    # Optional quality-of-life
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--device", default=None,
                   help="Device override (e.g., 'cuda', 'cpu', 'mps'). If omitted, auto-detect.")
    p.add_argument("--skip-existing", action="store_true",
                   help="If set, do not recompute when output file already exists.")
    p.add_argument("--limit", type=int, default=None,
                   help="If set, only process first N entries (useful for smoke tests).")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    device = args.device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

    log(f"Job started. Device: {device}")

    # Ensure output directory exists
    os.makedirs(args.concept_dir, exist_ok=True)

    # Load model
    log(f"Loading model: {args.model}")
    model = HookedTransformer.from_pretrained(args.model, device=device)

    # Load concept data
    log(f"Loading concept data from: {args.concept_data}")
    with open(args.concept_data, "r") as f:
        concept_data = json.load(f)

    # Main loop
    processed = 0
    with torch.no_grad():
        for idx, entry in enumerate(concept_data):
            if args.limit is not None and processed >= args.limit:
                log(f"Reached limit={args.limit}. Stopping.")
                break

            # Validate entry
            try:
                layer = entry["layer"]
                h_row = entry["h_row"]
                positive_sentences = entry["activating_sentences"]
                negative_sentences = entry["neutral_sentences"]
            except KeyError as e:
                log(f"Sample {idx} missing key {e}. Skipping.")
                continue

            fname = f"concept_l{layer}_h{h_row}.pt"
            save_fpath = os.path.join(args.concept_dir, fname)

            if args.skip_existing and os.path.isfile(save_fpath):
                log(f"Sample {idx}: {fname} exists, skipping.")
                continue

            log(f"Processing sample {idx}: layer={layer}, h_row={h_row}")

            # Compute concept vector
            dm = DiffMean(model, layer, model.tokenizer, device, mode=args.mode)
            dm.fit(positive_sentences, negative_sentences)
            concept_vector = dm.concept_vector  # tensor [hidden_dim]

            # Save
            torch.save(concept_vector.cpu(), save_fpath)
            log(f"Saved concept vector to {save_fpath}")

            processed += 1
            del dm
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    log("Job completed.")
