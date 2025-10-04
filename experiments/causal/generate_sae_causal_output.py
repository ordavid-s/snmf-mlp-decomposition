# experiments/sae_intervene_cli.py
import sys, os, json, random, argparse
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch

# keep project imports working when called as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformer_lens import HookedTransformer
from evaluation.json_handler import JsonHandler
from intervention.intervener import Intervener
from sae_lens import SAE


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

def parse_float_list(spec: str) -> List[float]:
    """
    Parse comma-separated floats: "0.025,0.05,0.1"
    """
    return [float(x.strip()) for x in spec.split(",") if x.strip()]

def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Intervene with SAE features and log causal effects."
    )

    # Required paths / model config
    parser.add_argument("--concept-json", required=True,
                        help="Path to the concept data JSON (entries with 'layer','index','sae_lens_release','sae_lens_id', ...).")
    parser.add_argument("--save-json", required=True,
                        help="Where to write results JSON (will append rows; file created if missing).")
    parser.add_argument("--model-name", required=True,
                        help='HF identifier for HookedTransformer, e.g. "gpt2-small" or "meta-llama/Llama-3.1-8B".')

    # Intervention configuration
    parser.add_argument("--intervention-type", default="mlp_out",
                        choices=["mlp_out","resid_pre","resid_mid","resid_post","attn_out"],
                        help="Where to inject the vector. Default: mlp_out.")
    parser.add_argument("--base-prompt", required=True,
                        help='Prompt to base KL search and generations on, e.g. "I think that".')
    parser.add_argument("--target-kls", required=True,
                        help='Comma-separated float targets for KL search, e.g. "0.025,0.05,0.1,0.15,0.25,0.35,0.5".')

    # Generation & logging
    parser.add_argument("--num-top-logits", type=int, required=True,
                        help="How many top changed logits to save per sign (+/-).")
    parser.add_argument("--num-sentences", type=int, required=True,
                        help="How many samples to generate per sign (+/-) per KL target.")
    parser.add_argument("--gen-max-new", type=int, default=50,
                        help="Max new tokens when sampling (default: 50).")
    parser.add_argument("--gen-top-k", type=int, default=30,
                        help="top_k sampling (default: 30).")
    parser.add_argument("--gen-top-p", type=float, default=0.3,
                        help="top_p sampling (default: 0.3).")

    # Execution environment
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--device", choices=["auto","cpu","cuda","mps"], default="auto",
                        help='Device choice (default: auto). "auto" picks cuda>mps>cpu.')

    # Optional filtering (process subset of layers or SAE ids if desired)
    parser.add_argument("--include-layers", default=None,
                        help='Optional comma-separated int list to restrict to specific layers, e.g. "0,6,12".')

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
    log(f"Device: {device}")

    # IO
    concept_json = args.concept_json
    save_json = args.save_json
    ensure_parent_dir(save_json)

    # Parse lists
    target_kls = parse_float_list(args.target_kls)
    include_layers = None
    if args.include_layers:
        include_layers = [int(x.strip()) for x in args.include_layers.split(",") if x.strip()]

    # Load concept data
    with open(concept_json, "r") as f:
        concept_data = json.load(f)

    # Group by SAE identity (layer, release, id)
    grouped = {}
    for entry in concept_data:
        layer = int(entry["layer"])
        if include_layers is not None and layer not in include_layers:
            continue
        key = (layer, entry["sae_lens_release"], entry["sae_lens_id"])
        grouped.setdefault(key, []).append(entry)

    log("Loading model…")
    model = HookedTransformer.from_pretrained(args.model_name, device=device)
    intervener = Intervener(model, intervention_type=args.intervention_type)

    json_handler = JsonHandler(
        ["layer", "h_row", "kl", "alpha", "top_logit_values", "top_shifted_tokens",
         "steered_sentences", "intervention_sign"],
        save_json,
        auto_write=False
    )

    base_prompt = args.base_prompt
    num_top = args.num_top_logits
    m_samples = args.num_sentences

    with torch.no_grad():
        # base logits once (prompt only)
        base_logits = model(model.to_tokens(base_prompt))

        for (layer, sae_release, sae_id), entries in grouped.items():
            log(f"Loading SAE — layer={layer}, release={sae_release}, id={sae_id}")
            # SAE.from_pretrained returns (sae, cfg) tuple; take sae
            sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)[0]

            for entry in entries:
                h_idx = int(entry["index"])
                concept_vec = sae.W_dec[h_idx, :]  # (d_model,) vector in model space

                # Find alpha matching KL targets (positive direction)
                kl_to_alpha = intervener.find_alpha_for_kl_targets(
                    base_prompt,
                    intervention_vectors=[concept_vec],
                    layers=[layer],
                    target_kls=target_kls
                )

                log(f"Entry index={h_idx} | generating for {len(kl_to_alpha)} KL targets")

                for kl, alpha in kl_to_alpha.items():
                    # +alpha intervention
                    logits_pos = intervener.intervene(
                        base_prompt, [concept_vec], layers=[layer], alpha=alpha
                    )
                    delta_pos = (logits_pos[0, -1, :] - base_logits[0, -1, :]).abs()
                    top_vals_pos, top_ids_pos = torch.topk(delta_pos, k=num_top)

                    # Convert token ids -> string tokens
                    top_tokens_pos = []
                    for tid in top_ids_pos:
                        # returns ["Ġword"] list; keep list for parity with original behavior
                        toks = model.to_str_tokens(tid)
                        top_tokens_pos.append(toks)

                    # -alpha intervention
                    logits_neg = intervener.intervene(
                        base_prompt, [concept_vec], layers=[layer], alpha=-alpha
                    )
                    delta_neg = (logits_neg[0, -1, :] - base_logits[0, -1, :]).abs()
                    top_vals_neg, top_ids_neg = torch.topk(delta_neg, k=num_top)

                    top_tokens_neg = []
                    for tid in top_ids_neg:
                        toks = model.to_str_tokens(tid)
                        top_tokens_neg.append(toks)

                    # Sample generations
                    s_pos = intervener.generate_with_manipulation_sampling(
                        base_prompt,
                        [concept_vec],
                        [layer],
                        alpha=alpha,
                        max_new_tokens=args.gen_max_new,
                        top_k=args.gen_top_k,
                        top_p=args.gen_top_p,
                        m=m_samples
                    )
                    s_neg = intervener.generate_with_manipulation_sampling(
                        base_prompt,
                        [concept_vec],
                        [layer],
                        alpha=-alpha,
                        max_new_tokens=args.gen_max_new,
                        top_k=args.gen_top_k,
                        top_p=args.gen_top_p,
                        m=m_samples
                    )

                    # Write rows
                    json_handler.add_row(
                        layer=layer,
                        h_row=h_idx,
                        alpha=float(alpha),
                        kl=float(kl),
                        top_logit_values=top_vals_pos.tolist(),
                        top_shifted_tokens=top_tokens_pos,
                        steered_sentences=s_pos,
                        intervention_sign=1,
                    )
                    json_handler.add_row(
                        layer=layer,
                        h_row=h_idx,
                        alpha=float(-alpha),
                        kl=float(kl),
                        top_logit_values=top_vals_neg.tolist(),
                        top_shifted_tokens=top_tokens_neg,
                        steered_sentences=s_neg,
                        intervention_sign=-1,
                    )
                    json_handler.write()

            # free GPU memory between SAE groups
            del sae
            if device == "cuda":
                torch.cuda.empty_cache()

    log("Done.")


if __name__ == "__main__":
    log("Job started.")
    main()
