import sys
import os
import argparse
import json
import pickle
from datetime import datetime

import torch
from transformer_lens import HookedTransformer
from factorization.seminmf import NMFSemiNMF
from experiments.evaluation.concept_evaluator import ConceptEvaluator
from experiments.evaluation.json_handler import JsonHandler


def log(txt: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {txt}", flush=True)


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def get_device(arg_device: str) -> str:
    if arg_device and arg_device.lower() != "auto":
        return arg_device
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate NMF concept vectors on sentences (paths must be provided explicitly)."
    )
    # Core experiment config (non-path)
    parser.add_argument("--mode", type=str, default="mlp",
                        help="Feature space to evaluate (e.g., mlp).")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B",
                        help="HF model name for HookedTransformer.")
    parser.add_argument("--layers", type=parse_int_list, default=[0, 6, 12, 18, 25, 31],
                        help='Comma-separated layer indices, e.g. "0,6,12,18,25,31".')
    parser.add_argument("--k-values", type=parse_int_list, default=[100],
                        help='Comma-separated K (ranks), e.g. "100" or "64,128".')
    parser.add_argument("--sparsity", type=str, default="s0.1",
                        help="Sparsity tag to store along with results (metadata only).")

    # 🔒 Required paths (no defaults)
    parser.add_argument("--save-path", required=True,
                        help="Path to write evaluation JSON (will be created/overwritten).")
    parser.add_argument("--concept-data", required=True,
                        help="Path to input JSON with generated/neutral sentences.")
    parser.add_argument("--models-root", required=True,
                        help="Root folder that contains per-K subfolders with NMF pickles.")

    # Devices
    parser.add_argument("--device", type=str, default="auto",
                        help='Device to use: "auto", "cuda", "cpu", or "mps".')
    parser.add_argument("--data-device", type=str, default="cpu",
                        help='Device for loading ancillary data (usually "cpu").')

    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    # Validate inputs
    if not os.path.isfile(args.concept_data):
        raise FileNotFoundError(f"--concept-data not found: {args.concept_data}")
    if not os.path.isdir(args.models_root):
        raise NotADirectoryError(f"--models-root not found: {args.models_root}")
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    # Load model and evaluator
    model = HookedTransformer.from_pretrained(args.model_name, device=device)
    evaluator = ConceptEvaluator(model)

    # JSON handler setup
    json_handler = JsonHandler(
        ["K", "concept", "scores", "random_scores", "layer", "h_row", "sparsity"],
        args.save_path,
        auto_write=False
    )

    # Load concept entries
    with open(args.concept_data) as f:
        concept_data = json.load(f)

    for K in args.k_values:
        log(f"Starting evaluation for K={K}...")


        # Load per-layer NMF models
        nmf_models = {}
        for layer in args.layers:
            factor_dir = os.path.join(args.models_root, str(layer), f"{K}")
            if not os.path.isdir(factor_dir):
                log(f"Dir does not exist, skipping K={K}: {factor_dir}")
                continue
            fn = f"nmf-l{layer}-r{K}.pkl"
            fp = os.path.join(factor_dir, fn)
            if os.path.isfile(fp):
                with open(fp, 'rb') as nmf_file:
                    nmf_models[layer] = pickle.load(nmf_file)
                log(f"Loaded NMF for layer {layer}, K={K}")
            else:
                log(f"Missing NMF file for layer {layer}, K={K}, skipping")

        # Evaluate each concept entry
        for entry in concept_data:
            if int(entry.get('K', -1)) != K:
                continue
            layer = entry['layer']
            nmf: NMFSemiNMF = nmf_models.get(layer)
            if nmf is None:
                continue

            h_row = entry['h_row']
            sentences = entry['activating_sentences']
            neutral_sentences = entry['neutral_sentences']
            concept = entry['concept']

            # normalized concept vector from F
            concept_vec = (nmf.F_.T[h_row] / nmf.F_.T[h_row].norm()).to(device)

            scores = evaluator.evaluate_tensor(sentences, layer, concept_vec)
            random_scores = evaluator.evaluate_tensor(neutral_sentences, layer, concept_vec)

            json_handler.add_row(
                K=K,
                concept=concept,
                scores=scores,
                random_scores=random_scores,
                layer=layer,
                h_row=h_row,
                sparsity=args.sparsity
            )
            del concept_vec

        # Clean for next K
        del nmf_models

    json_handler.write()
    log("Evaluation finished.")


if __name__ == "__main__":
    # Ensure repo root is on sys.path (one level up from this file)
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    main()
