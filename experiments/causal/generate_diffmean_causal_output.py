import sys
import os
import json
import random
import numpy as np
import torch
from datetime import datetime
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformer_lens import HookedTransformer
from evaluation.json_handler import JsonHandler
from intervention.intervener import Intervener


# ────────────────────────── utils ──────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log(txt):
    print(f"[{datetime.now()}] {txt}", flush=True)


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


# ─────────────────────────── main ───────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run causal diffmean evaluation using saved concept vectors"
    )

    # Required core config (no defaults)
    parser.add_argument("--model-name", required=True, type=str,
                        help="HF model name for HookedTransformer (e.g., 'meta-llama/Llama-3.1-8B' or 'gemma-2-2b').")
    parser.add_argument("--mode", required=True, type=str,
                        help="Intervention mode (passed to Intervener as intervention_type).")
    parser.add_argument("--base-path", required=True, type=str,
                        help="Base project path. Other templates are joined to this.")
    parser.add_argument("--vectors-dir-tpl", required=True, type=str,
                        help="Template for vectors directory (must include '{mode}' if you want it interpolated).")
    parser.add_argument("--save-path-tpl", required=True, type=str,
                        help="Template for output JSON path (must include '{mode}' if you want it interpolated).")
    parser.add_argument("--data-path", required=True, type=str,
                        help="Path to input concept metadata JSON.")
    parser.add_argument("--layers", required=True, type=str,
                        help="Comma-separated list of layers to process (e.g., '0,6,12,18,25').")
    parser.add_argument("--target-kls", required=True, type=str,
                        help="Comma-separated list of target KLs (e.g., '0.025,0.05,0.1').")
    parser.add_argument("--num-top", required=True, type=int,
                        help="Top-K deltas to record per direction.")
    parser.add_argument("--num-sent", required=True, type=int,
                        help="Number of steered samples to generate per sign.")
    parser.add_argument("--base-prompt", required=True, type=str,
                        help="Base prompt used for interventions.")

    # Optional knobs
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu", "mps"], default="auto",
                        help="Device override. 'auto' picks CUDA->MPS->CPU.")
    parser.add_argument("--factor-function", type=str, default=None,
                        help="Name/selector for a factor function if your Intervener supports it (e.g., 'gemma' or 'default').")
    parser.add_argument("--gen-max-new-tokens", type=int, default=50, help="Generation: max_new_tokens.")
    parser.add_argument("--gen-top-k", type=int, default=30, help="Generation: top_k.")
    parser.add_argument("--gen-top-p", type=float, default=0.3, help="Generation: top_p.")

    args = parser.parse_args()

    # Seed & device
    set_seed(args.seed)
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    mode = args.mode
    model_name = args.model_name

    base_path = args.base_path
    vectors_dir = os.path.join(base_path, args.vectors_dir_tpl.format(mode=mode))
    save_path   = os.path.join(base_path, args.save_path_tpl.format(mode=mode))
    data_path   = os.path.join(base_path, args.data_path)

    layers = parse_int_list(args.layers)
    target_kls = parse_float_list(args.target_kls)
    required = len(target_kls) * 2  # pos+neg per KL

    log("Job started.")
    log(f"Model: {model_name} | Mode: {mode} | Device: {device}")
    log(f"Vectors dir: {vectors_dir}")
    log(f"Save path:   {save_path}")
    log(f"Data path:   {data_path}")
    log(f"Layers:      {layers}")
    log(f"Target KLs:  {target_kls}")
    log(f"Seed:        {args.seed}")

    # Track how many rows exist per (layer, h_row)
    processed_counts = {}
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
        for row in existing:
            key = (row.get('layer'), row.get('h_row'))
            processed_counts[key] = processed_counts.get(key, 0) + 1

    # Load model & Intervener
    model = HookedTransformer.from_pretrained(model_name, device=device)

    # Build Intervener (supporting optional factor_function if constructor takes it)
    try:
        intervener = Intervener(model, intervention_type=mode, factor_function=args.factor_function)
    except TypeError:
        intervener = Intervener(model, intervention_type=mode)

    # IO helpers
    json_handler = JsonHandler(
        ["layer", "h_row", "kl", "alpha",
         "top_logit_values", "top_shifted_tokens",
         "steered_sentences", "intervention_sign"],
        save_path,
        auto_write=False
    )

    # Load concept metadata
    with open(data_path, 'r') as f:
        concept_data = json.load(f)
    concept_data = [c for c in concept_data if int(c['layer']) in layers]
    log(f"Total concepts to process after layer filter: {len(concept_data)}")

    with torch.no_grad():
        base_logits = model(model.to_tokens(args.base_prompt))

        for idx, entry in enumerate(concept_data):
            layer = entry['layer']
            h_row = entry['h_row']
            key = (layer, h_row)
            count = processed_counts.get(key, 0)
            if count >= required:
                log(f"[{idx+1}/{len(concept_data)}] Skip l{layer}, h{h_row}: already {count} rows")
                continue

            path = os.path.join(vectors_dir, f"concept_l{layer}_h{h_row}.pt")
            if not os.path.isfile(path):
                log(f"[{idx+1}/{len(concept_data)}] Missing vector at {path}, skipping")
                continue

            log(f"[{idx+1}/{len(concept_data)}] Loading vector: {os.path.basename(path)}")
            concept_vector = torch.load(path, map_location=device)

            # compute alphas for each KL target
            try:
                kl_to_alpha = intervener.find_alpha_for_kl_targets(
                    args.base_prompt,
                    intervention_vectors=[concept_vector],
                    layers=[layer],
                    target_kls=target_kls,
                    factor_function=args.factor_function
                )
            except TypeError:
                kl_to_alpha = intervener.find_alpha_for_kl_targets(
                    args.base_prompt,
                    intervention_vectors=[concept_vector],
                    layers=[layer],
                    target_kls=target_kls
                )

            for kl, alpha in kl_to_alpha.items():
                # positive intervention
                try:
                    int_logits = intervener.intervene(
                        args.base_prompt, [concept_vector], [layer], alpha=alpha,
                        factor_function=args.factor_function
                    )
                except TypeError:
                    int_logits = intervener.intervene(
                        args.base_prompt, [concept_vector], [layer], alpha=alpha
                    )

                delta = (int_logits[0, -1, :] - base_logits[0, -1, :])
                top_vals, top_idxs = torch.topk(delta, k=args.num_top)
                top_vals = top_vals.tolist()
                top_toks = [model.to_str_tokens(i) for i in top_idxs]

                # negative intervention
                try:
                    int_logits_neg = intervener.intervene(
                        args.base_prompt, [concept_vector], [layer], alpha=-alpha,
                        factor_function=args.factor_function
                    )
                except TypeError:
                    int_logits_neg = intervener.intervene(
                        args.base_prompt, [concept_vector], [layer], alpha=-alpha
                    )

                delta_neg = (int_logits_neg[0, -1, :] - base_logits[0, -1, :])
                top_vals_neg, top_idxs_neg = torch.topk(delta_neg, k=args.num_top)
                top_vals_neg = top_vals_neg.tolist()
                top_toks_neg = [model.to_str_tokens(i) for i in top_idxs_neg]

                # generate steered sentences
                gen_kwargs = dict(
                    max_new_tokens=args.gen_max_new_tokens,
                    top_k=args.gen_top_k,
                    top_p=args.gen_top_p,
                    m=args.num_sent,
                )
                try:
                    s_pos = intervener.generate_with_manipulation_sampling(
                        args.base_prompt, [concept_vector], [layer], alpha=alpha,
                        factor_function=args.factor_function, **gen_kwargs
                    )
                    s_neg = intervener.generate_with_manipulation_sampling(
                        args.base_prompt, [concept_vector], [layer], alpha=-alpha,
                        factor_function=args.factor_function, **gen_kwargs
                    )
                except TypeError:
                    s_pos = intervener.generate_with_manipulation_sampling(
                        args.base_prompt, [concept_vector], [layer], alpha=alpha, **gen_kwargs
                    )
                    s_neg = intervener.generate_with_manipulation_sampling(
                        args.base_prompt, [concept_vector], [layer], alpha=-alpha, **gen_kwargs
                    )

                # write rows
                json_handler.add_row(
                    layer=layer, h_row=h_row,
                    alpha=alpha, kl=kl,
                    top_logit_values=top_vals,
                    top_shifted_tokens=top_toks,
                    steered_sentences=s_pos,
                    intervention_sign=1
                )
                json_handler.add_row(
                    layer=layer, h_row=h_row,
                    alpha=-alpha, kl=kl,
                    top_logit_values=top_vals_neg,
                    top_shifted_tokens=top_toks_neg,
                    steered_sentences=s_neg,
                    intervention_sign=-1
                )

            json_handler.write()
            if device == "cuda":
                torch.cuda.empty_cache()

    log("Job completed.")
