#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import random
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

from evaluation.json_handler import JsonHandler
from intervention.intervener import Intervener


# ------------------------------
# Utils
# ------------------------------
def log(txt: str) -> None:
    print(f"[{datetime.now()}] {txt}", flush=True)

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def get_vocab_proj(A: torch.Tensor, model: HookedTransformer, top_k: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project a direction to vocab logits and return top-k values+indices.
    Works on the model's device.
    """
    # move A to model device
    dev = next(model.parameters()).device
    A = A.to(dev)
    direction = model.ln_final(A)
    vocab_proj = model.unembed(direction)
    values, indices = torch.topk(vocab_proj, k=top_k)
    return values, indices


# ------------------------------
# Main
# ------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute top-vocab projections for SAE concept vectors and save to JSON."
    )

    # Required paths / files
    parser.add_argument("--concept-data", required=True,
                        help="Path to JSON with concept entries (must include 'layer','index','sae_lens_release','sae_lens_id').")
    parser.add_argument("--out-json", required=True,
                        help="Where to write the output JSON (vocab projections).")

    # Model / runtime
    parser.add_argument("--model-name", required=True,
                        help="HF identifier for HookedTransformer (e.g., 'gpt2-small', 'meta-llama/Llama-3.1-8B').")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu", "mps"],
                        help="Override device. Default: auto-detect.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # Processing options
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-K tokens to keep from vocab projection.")
    parser.add_argument("--intervention-type", default="mlp_out",
                        help="Intervention space for Intervener (kept for parity with your flow).")
    parser.add_argument("--auto-write", action="store_true",
                        help="If set, JsonHandler writes on every add_row; otherwise writes once per group.")

    # Optional filtering
    parser.add_argument("--only-layers", default=None,
                        help="Optional layer filter like '0,6,12' or '0-3'.")
    parser.add_argument("--only-indices", default=None,
                        help="Optional neuron index filter like '5,12,20-25' (applied within each SAE).")

    args = parser.parse_args()

    # Resolve device
    if args.device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    log(f"Device: {device}")
    set_seed(args.seed)

    # Load input concepts
    with open(args.concept_data, "r") as f:
        concept_data = json.load(f)

    # Optional filters
    def parse_int_list(spec: str) -> List[int]:
        if not spec:
            return []
        out = []
        parts = [p.strip() for p in spec.split(",")]
        for p in parts:
            if "-" in p:
                a, b = p.split("-", 1)
                out.extend(list(range(int(a), int(b) + 1)))
            else:
                out.append(int(p))
        return sorted(set(out))

    layer_filter = set(parse_int_list(args.only_layers)) if args.only_layers else None
    index_filter = set(parse_int_list(args.only_indices)) if args.only_indices else None

    # Group by SAE identity (layer, release, id)
    grouped: Dict[Tuple[int, str, str], List[dict]] = {}
    for entry in concept_data:
        layer = int(entry["layer"])
        if layer_filter is not None and layer not in layer_filter:
            continue
        if index_filter is not None and int(entry["index"]) not in index_filter:
            continue
        key = (layer, entry["sae_lens_release"], entry["sae_lens_id"])
        grouped.setdefault(key, []).append(entry)

    if not grouped:
        raise RuntimeError("No entries to process after filtering—check your --only-layers / --only-indices.")

    # Init model + helpers
    model = HookedTransformer.from_pretrained(args.model_name, device=device)
    _ = Intervener(model, intervention_type=args.intervention_type)  # keeps parity with your flow

    json_handler = JsonHandler(
        ["layer", "h_row", "top_logit_values", "top_shifted_tokens", "intervention_sign"],
        args.out_json,
        auto_write=args.auto_write,
    )

    # Process each SAE group
    for (layer, sae_release, sae_id), entries in grouped.items():
        log(f"Loading SAE for layer={layer}, release={sae_release}, id={sae_id}")
        sae_instance = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)[0]

        # For each concept in this SAE, project +/- concept vector to vocab
        for entry in entries:
            h_idx = int(entry["index"])

            # SAE decoder vector for this neuron
            concept_vec = sae_instance.W_dec[h_idx, :].contiguous()

            # Positive direction
            pos_vals, pos_idx = get_vocab_proj(concept_vec, model, top_k=args.top_k)
            pos_vals_list = pos_vals.tolist()
            # Convert token IDs to strings; ensure int() cast for safety
            pos_tok_list = [model.to_str_tokens([tid])[0] for tid in pos_idx]
            print([model.to_str_tokens([tid]) for tid in pos_idx])

            json_handler.add_row(
                layer=layer,
                h_row=h_idx,
                top_logit_values=pos_vals_list,
                top_shifted_tokens=pos_tok_list,
                intervention_sign=1,
            )

            # Negative direction
            neg_vals, neg_idx = get_vocab_proj(-concept_vec, model, top_k=args.top_k)
            neg_vals_list = neg_vals.tolist()
            neg_tok_list = [model.to_str_tokens([tid])[0] for tid in neg_idx]

            json_handler.add_row(
                layer=layer,
                h_row=h_idx,
                top_logit_values=neg_vals_list,
                top_shifted_tokens=neg_tok_list,
                intervention_sign=-1,
            )

        log(f"Finished processing layer {layer}")
        if not args.auto_write:
            json_handler.write()

    # Final flush if auto_write was on (JsonHandler may no-op if already written)
    json_handler.write()
    log(f"Done. Wrote: {args.out_json}")


if __name__ == "__main__":
    main()
