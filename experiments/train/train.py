from llm_utils.activation_generator import ActivationGenerator
from factorization.seminmf import NMFSemiNMF
from data_utils.concept_dataset import SupervisedConceptDataset

import argparse
import random
import numpy as np
import torch
import pickle
from pathlib import Path
from datetime import datetime
from typing import List

# ------------------------------
# Helpers
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

def parse_int_list(spec: str) -> List[int]:
    """
    Parse '0,1,2' or '0-3' or '0,2,5-7' into a list of ints.
    """
    out = []
    for chunk in spec.split(','):
        chunk = chunk.strip()
        if '-' in chunk:
            a, b = chunk.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        elif chunk:
            out.append(int(chunk))
    return sorted(set(out))

def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run Semi-NMF with different initialization methods over model layer activations."
    )
    # Initialization & training knobs
    parser.add_argument("--sparsity", type=float, default=0.01, help="L1 sparsity weight for Semi-NMF.")
    parser.add_argument("--ranks", type=str, default="100",
                        help="Comma/range list of ranks, e.g. '50,100' or '32-64'.")
    parser.add_argument("--max-iterations-per-layer", type=int, default=200_000,
                        help="Max optimization steps per layer.")
    parser.add_argument("--patience", type=int, default=1500,
                        help="Early-stopping patience (steps).")
    parser.add_argument("--overwrite", action="store_true",
                        help="If set, refit and overwrite existing .pkl files.")

    # Model & data
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B",
                        help="HF model name for ActivationGenerator.")
    parser.add_argument("--factorization-mode", type=str, default="mlp",
                        help="Activation mode for ActivationGenerator (e.g., 'mlp').")
    parser.add_argument("--layers", type=str, default="0-31",
                        help="Comma/range list of layers, e.g. '0-31' or '0,4,10-12'.")
    parser.add_argument("--data-path", type=str, default="data/final_dataset_20_concepts.json",
                        help="Path to the supervised concepts JSON.")

    # Devices
    parser.add_argument("--model-device", type=str, default=default_device(),
                        help="Device for the model (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--data-device", type=str, default="cpu",
                        help="Device for holding/generated activations.")
    parser.add_argument("--fitting-device", type=str, default=default_device(),
                        help="Device for NMF fitting (usually matches --model-device).")

    # Paths
    parser.add_argument("--base-path", type=str, default=".",
                        help="Base path used to resolve defaults for save paths.")
    parser.add_argument("--save-path", type=str, default=None,
                        help="Where to save models. If omitted, uses {base}/rebuttal/init_exp/models/{init}.")

    # Repro & misc
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    args = parser.parse_args()

    # Resolve parsed lists
    layers = parse_int_list(args.layers)
    ranks = parse_int_list(args.ranks)

    # Seed & devices
    set_seed(args.seed)
    model_device = args.model_device
    data_device = args.data_device
    fitting_device = args.fitting_device

    # Paths
    base_path = Path(args.base_path).resolve()
    if args.save_path is None:
        save_path = base_path / "rebuttal" / "init_exp" / "models"
    else:
        save_path = Path(args.save_path).resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path).resolve()

    # Config summary
    log("Job started.")
    log(f"""
==== Configuration Summary ====
Base Path:             {base_path}
Save Path:             {save_path}
Data Path:             {data_path}
Model Name:            {args.model_name}
Layers to Inspect:     {layers}
Ranks:                 {ranks}
Sparsity:              {args.sparsity}
Factorization Mode:    {args.factorization_mode}
Model Device:          {model_device}
Data Device:           {data_device}
Fitting Device:        {fitting_device}
Max Iters / Layer:     {args.max_iterations_per_layer}
Patience:              {args.patience}
Overwrite Existing:    {args.overwrite}
Seed:                  {args.seed}
===============================
""".strip())

    # Generate activations
    log(f"Initializing ActivationGenerator with model '{args.model_name}'")
    act_generator = ActivationGenerator(
        args.model_name,
        model_device=model_device,
        data_device=data_device,
        mode=args.factorization_mode,
    )

    log(f"Loading dataset from '{data_path}'")
    dataset = SupervisedConceptDataset(str(data_path))

    log("Generating activations (and frequencies) for requested layers...")
    activations, freq = act_generator.generate_multiple_layer_activations_and_freq(dataset, layers)

    # Factorize per-layer
    for idx, layer in enumerate(layers):
        log(f"Processing layer {layer}")

        layer_dir = save_path / str(layer)
        layer_dir.mkdir(parents=True, exist_ok=True)

        for rank in ranks:
            file_path = layer_dir / str(rank) / f"nmf-l{layer}-r{rank}.pkl"
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if file_path.exists() and not args.overwrite:
                log(f"File exists, skipping: {file_path}")
                continue

            nmf = NMFSemiNMF(rank=rank, fitting_device=fitting_device, sparsity=args.sparsity)
            log(f"Fitting Semi-NMF for layer {layer}, rank {rank}")
            nmf.fit(
                (activations[idx]).T,
                args.max_iterations_per_layer,
                patience=args.patience,
                init="random",
            )

            with open(file_path, "wb") as f:
                pickle.dump(nmf, f)
            log(f"Saved model → {file_path}")

            del nmf
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                log("Emptied CUDA cache")

    log("All computations done.")

if __name__ == "__main__":
    main()
