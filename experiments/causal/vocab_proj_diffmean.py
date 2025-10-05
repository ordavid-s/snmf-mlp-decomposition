import sys
import os
import json
import random
import argparse
from datetime import datetime

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformer_lens import HookedTransformer
from evaluation.json_handler import JsonHandler


# ────────────────────────── utils ──────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log(txt: str):
    print(f"[{datetime.now()}] {txt}", flush=True)


def parse_int_list(spec: str):
    """
    Parse '0,1,2' or '0-3' or '0,2,5-7' into a sorted list of ints.
    """
    out = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            a, b = int(a), int(b)
            lo, hi = (a, b) if a <= b else (b, a)
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return sorted(out)


@torch.no_grad()
def get_vocab_proj(A: torch.Tensor, model: HookedTransformer, top_k: int = 50):
    """
    Returns top_k (values, indices) for the unembed(logits) produced from A after final layernorm.
    Assumes A is on the same device as the model.
    """
    # Ensure same device
    A = A.to(next(model.parameters()).device)
    direction = model.ln_final(A)
    vocab_proj = model.unembed(direction)          # (vocab,)
    values, indices = torch.topk(vocab_proj, top_k)
    return values, indices


def main():
    parser = argparse.ArgumentParser(
        description="Compute top shifted tokens for saved concept vectors and write to JSON."
    )
    # Core behavior
    parser.add_argument("--mode", type=str, required=True,
                        help="Intervention/factoring mode string (e.g., mlp_out).")
    parser.add_argument("--model-name", type=str, required=True,
                        help='Model name for HookedTransformer.from_pretrained (e.g., "gemma-2-2b").')

    # Paths (explicit; no hidden defaults)
    parser.add_argument("--vectors-dir", type=str, required=True,
                        help="Directory containing concept vectors (*.pt) named like concept_l{layer}_h{h_row}.pt")
    parser.add_argument("--data-path", type=str, required=True,
                        help="JSON file with concept metadata (expects keys: 'layer', 'h_row').")
    parser.add_argument("--save-path", type=str, required=True,
                        help="Output JSON path to write results to.")

    # Selection / knobs
    parser.add_argument("--layers", type=str, required=True,
                        help="Layers filter, e.g. '18,25' or '0-3' or '0,2,5-7'.")
    parser.add_argument("--top-k", type=int, default=50,
                        help="How many top logits/tokens to keep per sign. Default: 50")

    # Runtime
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu", "mps"], default="auto",
                        help="Force device. Default: auto (cuda→mps→cpu).")

    args = parser.parse_args()

    # Device resolve
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    set_seed(args.seed)
    log("Job started.")
    log(f"Mode: {args.mode}")
    log(f"Model: {args.model_name if hasattr(args,'model-name') else args.model_name}")  # safeguard for shells
    log(f"Device: {device}")

    # Load model
    model = HookedTransformer.from_pretrained(args.model_name, device=device)

    # Prepare output writer
    num_top_logits_to_save = int(args.top_k)
    json_handler = JsonHandler(
        ["layer", "h_row", "top_logit_values", "top_shifted_tokens", "intervention_sign"],
        args.save_path,
        auto_write=False
    )

    # Load and filter metadata
    with open(args.data_path, "r") as f:
        concept_data = json.load(f)

    layers = set(parse_int_list(args.layers))
    concept_data = [c for c in concept_data if int(c["layer"]) in layers]

    log(f"Processing {len(concept_data)} concepts across layers {sorted(layers)}")

    with torch.no_grad():
        for idx, entry in enumerate(concept_data):
            layer = int(entry["layer"])
            h_row = int(entry["h_row"])
            vector_path = os.path.join(args.vectors_dir, f"concept_l{layer}_h{h_row}.pt")

            if not os.path.isfile(vector_path):
                log(f"[{idx+1}/{len(concept_data)}] Missing vector at {vector_path}, skipping.")
                continue

            log(f"[{idx+1}/{len(concept_data)}] Loading vector: {os.path.basename(vector_path)}")
            concept_vector = torch.load(vector_path, map_location=device)

            # Positive direction
            pos_vals_t, pos_ids_t = get_vocab_proj(concept_vector, model, top_k=num_top_logits_to_save)
            # Negative direction
            neg_vals_t, neg_ids_t = get_vocab_proj(-concept_vector, model, top_k=num_top_logits_to_save)

            # Convert to Python lists
            pos_vals = pos_vals_t.detach().cpu().tolist()
            neg_vals = neg_vals_t.detach().cpu().tolist()
            pos_ids = [x for x in pos_ids_t.detach().cpu()]
            neg_ids = [x for x in neg_ids_t.detach().cpu()]

            # Convert token ids → strings (avoid BOS)
            pos_toks = [model.to_str_tokens(tid, prepend_bos=False)[0] for tid in pos_ids]
            neg_toks = [model.to_str_tokens(tid, prepend_bos=False)[0] for tid in neg_ids]

            # Write rows
            json_handler.add_row(
                layer=layer,
                h_row=h_row,
                top_logit_values=pos_vals,
                top_shifted_tokens=pos_toks,
                intervention_sign=1,
            )
            json_handler.add_row(
                layer=layer,
                h_row=h_row,
                top_logit_values=neg_vals,
                top_shifted_tokens=neg_toks,
                intervention_sign=-1,
            )

            del concept_vector
            json_handler.write()  # persist incrementally

    log("Job completed.")


if __name__ == "__main__":
    main()
