# run_causal_from_nmf.py
import sys
import os
import json
import argparse
import pickle
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import numpy as np
import torch
from tqdm import tqdm

from transformer_lens import HookedTransformer
from evaluation.json_handler import JsonHandler
from intervention.intervener import Intervener
from factorization.seminmf import NMFSemiNMF

# Gemma-only utility (safe to import; only used if gemma)
try:
    from transformer_lens.utilities.addmm import batch_addmm
except Exception:
    batch_addmm = None


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

def parse_int_list(spec: str):
    """
    Parse '0,1,2' or '0-3' or '0,2,5-7' into a list of ints.
    """
    out = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out

def is_gemma_model(model_name: str) -> bool:
    return "gemma" in model_name.lower()

def get_concept_vector_regular(nmf_F_row: torch.Tensor) -> torch.Tensor:
    # unit-normalized factor row (as in your "regular" script)
    v = nmf_F_row
    return v / (v.norm() + 1e-12)

def get_concept_vector_gemma(mlp_vec: torch.Tensor, model: HookedTransformer, layer: int, device: torch.device) -> torch.Tensor:
    """
    Gemma: map an MLP-out direction into the logits space with the same path used by generation-time
    interventions: y = ln2_post( b_out + W_out @ mlp_vec )
    """
    if batch_addmm is None:
        raise RuntimeError("batch_addmm not available. Ensure transformer_lens >= 2.4 and correct install.")
    W = model.W_out[layer]              # (d_model, d_mlp)
    b = model.b_out[layer]              # (d_model,)
    y = batch_addmm(b, W, mlp_vec.to(device))
    return model.blocks[layer].ln2_post(y)


# ------------------------------
# Main
# ------------------------------
def main():
    set_seed(42)

    parser = argparse.ArgumentParser(
        description="Run causal interventions from NMF factors; auto-select Gemma vs Regular vector construction."
    )
    # Required core knobs
    parser.add_argument("--model-name", required=True, type=str,
                        help="HF repo id used by HookedTransformer (e.g., 'meta-llama/Llama-3.1-8B' or 'gemma-2-2b').")
    parser.add_argument("--layers", required=True, type=parse_int_list,
                        help="Layers to use, e.g. '0,6,12,18,25' or '0-4'.")
    parser.add_argument("--ranks", required=True, type=parse_int_list,
                        help="Ranks K to iterate, e.g. '100' or '50,100'.")
    parser.add_argument("--factorization-base-path", required=True, type=str,
                        help="Base directory where factorization models live and outputs are written.")
    parser.add_argument("--save-path", required=True, type=str,
                        help="Base directory where factorization models live and outputs are written.")
    parser.add_argument("--sparsity", required=True, type=str,
                        help="Sparsity identifier used in your run naming (string; used in paths/filenames).")
    # Optional overrides
    parser.add_argument("--num-top-logits", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-sentences", type=int, default=8)
    parser.add_argument("--base-prompt", type=str, default="I think that")
    parser.add_argument("--target-kls", type=str, default="0.025,0.05,0.1,0.15,0.25,0.35,0.5",
                        help="Comma-separated list of KL targets.")

    args = parser.parse_args()
    device = args.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model_name = args.model_name
    layers = args.layers
    ranks = args.ranks
    sparsity = args.sparsity
    num_top_logits_to_save = args.num_top_logits
    num_sentences_to_generate = args.num_sentences
    base_prompt = args.base_prompt
    target_kls = [float(x.strip()) for x in args.target_kls.split(",") if x.strip()]

    gemma = is_gemma_model(model_name)

    factorization_base_path = args.factorization_base_path
    save_path = args.save_path

    # Track how many rows we already created per (layer, h_row, K)
    processed_counts = {}
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
        for row in existing:
            key = (row.get("layer"), row.get("h_row"), row.get("K"))
            processed_counts[key] = processed_counts.get(key, 0) + 1

    log("Loading model…")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    intervener_kwargs = {"intervention_type": "mlp_out"} if gemma else {}
    intervener = Intervener(model, **intervener_kwargs)

    json_handler = JsonHandler(
        ["K", "layer", "h_row", "alpha", "kl", "top_logit_values", "top_shifted_tokens", "steered_sentences", "intervention_sign", "sparsity"],
        save_path,
        auto_write=False
    )

    required_rows = len(target_kls) * 2  # pos+neg

    with torch.no_grad():
        base_logits = model(model.to_tokens(base_prompt))

    for rank in ranks:
        log(f"\n--- Rank {rank} ---")
        # load per-layer NMFs
        nmf_models = {}
        for layer in layers:
            rank_dir = os.path.join(factorization_base_path, str(layer), str(rank))
            if not os.path.isdir(rank_dir):
                log(f"Dir does not exist, skipping rank: {rank}")
                continue

            fn = f"nmf-l{layer}-r{rank}.pkl"
            fp = os.path.join(rank_dir, fn)
            if os.path.isfile(fp):
                with open(fp, "rb") as f:
                    nmf_models[layer] = pickle.load(f)
                log(f"Loaded NMF model for layer {layer}, rank {rank}")
            else:
                log(f"Missing NMF file for layer {layer}, rank {rank} → skipping")

        # intervention flow
        for layer in layers:
            nmf: NMFSemiNMF = nmf_models.get(layer)
            if nmf is None:
                continue

            # nmf.F_.T[h_idx] is the factor row (dimension depends on space)
            for h_idx in tqdm(range(rank)):
                key = (layer, h_idx, rank)
                if processed_counts.get(key, 0) >= required_rows:
                    log(f"Skipping layer {layer}, h_row {h_idx}: already has {processed_counts[key]} rows.")
                    continue

                with torch.no_grad():
                    if gemma:
                        # Gemma branch: map MLP-out factor via W_out/b_out + LN2_post
                        factor_row = nmf.F_.T[h_idx].to(device)
                        concept_vector = get_concept_vector_gemma(factor_row, model, layer, device)
                    else:
                        # Regular branch: unit-normalized factor row
                        concept_vector = get_concept_vector_regular(nmf.F_.T[h_idx].to(device))

                    # Find alpha per KL
                    kl_to_alpha = intervener.find_alpha_for_kl_targets(
                        base_prompt,
                        intervention_vectors=[concept_vector],
                        layers=[layer],
                        target_kls=target_kls
                    )

                    for kl, alpha in kl_to_alpha.items():
                        intervened_logits = intervener.intervene(
                            base_prompt,
                            [concept_vector],
                            layers=[layer],
                            alpha=alpha
                        )
                        delta = (intervened_logits[0, -1, :] - base_logits[0, -1, :])
                        # positive shifts
                        pos_vals, pos_ids = torch.topk(delta, k=num_top_logits_to_save)
                        pos_list = pos_vals.tolist()
                        pos_tokens = [model.to_str_tokens(tid) for tid in pos_ids]

                        # negative shifts
                        neg_vals, neg_ids = torch.topk(-delta, k=num_top_logits_to_save)
                        neg_list = neg_vals.tolist()
                        neg_tokens = [model.to_str_tokens(tid) for tid in neg_ids]

                        # generate steered sentences
                        sentences_pos = intervener.generate_with_manipulation_sampling(
                            base_prompt,
                            [concept_vector],
                            [layer],
                            alpha=alpha,
                            max_new_tokens=50,
                            top_k=30,
                            top_p=0.3,
                            m=num_sentences_to_generate
                        )
                        sentences_neg = intervener.generate_with_manipulation_sampling(
                            base_prompt,
                            [concept_vector],
                            [layer],
                            alpha=-alpha,
                            max_new_tokens=50,
                            top_k=30,
                            top_p=0.3,
                            m=num_sentences_to_generate
                        )

                        # write rows
                        json_handler.add_row(
                            K=rank,
                            layer=layer,
                            h_row=h_idx,
                            alpha=alpha,
                            kl=kl,
                            top_logit_values=pos_list,
                            top_shifted_tokens=pos_tokens,
                            steered_sentences=sentences_pos,
                            intervention_sign=1,
                            sparsity=sparsity
                        )
                        json_handler.add_row(
                            K=rank,
                            layer=layer,
                            h_row=h_idx,
                            alpha=alpha,
                            kl=kl,
                            top_logit_values=neg_list,
                            top_shifted_tokens=neg_tokens,
                            steered_sentences=sentences_neg,
                            intervention_sign=-1,
                            sparsity=sparsity
                        )

                del concept_vector
                log(f"Finished row: layer={layer}, h_row={h_idx}")
            log(f"Finished layer: {layer}, rank: {rank}")
            json_handler.write()

        del nmf_models  # free memory

    log("Job finished.")


if __name__ == "__main__":
    main()
