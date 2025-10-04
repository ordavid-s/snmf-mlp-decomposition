import json
from pathlib import Path
import argparse
import torch

from transformer_lens import HookedTransformer
from experiments.evaluation.concept_evaluator import ConceptEvaluator
from experiments.evaluation.json_handler import JsonHandler
from sae_lens import SAE


def parse_layers(spec: str):
    """
    Parse layer specs like:
      '0,3,6,9,11' or '0-3,6-9' or mix '0-3,6,9-11'
    into a sorted list of unique ints.
    """
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    out = []
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            out.extend(range(a, b + 1))
        else:
            out.append(int(p))
    if not out:
        raise ValueError("No layers parsed from --layers.")
    return sorted(set(out))


def main():
    p = argparse.ArgumentParser(
        description="Evaluate SAE features on concept sentences across layers."
    )

    # All core configs are REQUIRED (no defaults).
    p.add_argument("--model-name", required=True,
                   help="Model name for HookedTransformer.from_pretrained (e.g., gpt2-small).")
    p.add_argument("--layers", required=True,
                   help="Layers to evaluate, e.g. '0,3,6,9,11' or '0-3,9-11'.")
    p.add_argument("--hook-template", required=True,
                   help="Hook template with '{layer_number}', e.g. 'blocks.{layer_number}.hook_mlp_out'.")
    p.add_argument("--concept-json", required=True,
                   help="Path to input JSON with concept sentences (expects top-level key 'results').")
    p.add_argument("--sentences-json", required=True,
                   help="Path to input JSON with generated sentences (expects top-level key 'results').")
    p.add_argument("--save-path", required=True,
                   help="Where to write evaluation results JSON.")
    p.add_argument("--device", required=True, choices=["cuda", "cpu", "mps"],
                   help="Compute device to use.")

    # Optional behavior flags (not core experiment settings)
    p.add_argument("--auto-write", action="store_true",
                   help="Write after each row is added (safer for long runs).")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output file.")
    p.add_argument("--verbose", action="store_true",
                   help="Print progress info.")

    args = p.parse_args()

    # Resolve paths
    concept_path = Path(args.concept_json)
    sentences_path = Path(args.sentences_json)
    out_path = Path(args.save_path)

    if not concept_path.is_file():
        raise FileNotFoundError(f"Concept JSON not found: {concept_path}")

    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {out_path} (use --overwrite to replace)")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate hook template
    if "{layer_number}" not in args.hook_template:
        raise ValueError("--hook-template must include '{layer_number}'")

    # Parse layers
    layers = parse_layers(args.layers)
    layer_set = set(layers)

    # Echo config (no defaults)
    if args.verbose:
        print(f"[CFG] model={args.model_name} device={args.device}")
        print(f"[CFG] layers={layers}")
        print(f"[CFG] hook_template='{args.hook_template}'")
        print(f"[CFG] concept_json={concept_path}")
        print(f"[CFG] save_path={out_path}")
        print(f"[CFG] auto_write={args.auto_write} overwrite={args.overwrite}")

    # Load model and helpers
    model = HookedTransformer.from_pretrained(args.model_name, device=args.device)
    evaluator = ConceptEvaluator(model, hook_template=args.hook_template)
    writer = JsonHandler(
        ["concept", "scores", "random_scores", "layer", "h_row"],
        str(out_path),
        auto_write=args.auto_write,
    )

    # Load concept data
    with concept_path.open() as f:
        concept_data = json.load(f)
    with sentences_path.open() as f:
        sentences_data = json.load(f)
    
    for e in sentences_data:
        for e2 in concept_data:
            if e['h_row'] == e2['index']:
                e.update(**e2)

    # Group by (layer, sae_lens_release, sae_lens_id)
    groups = {}
    for e in sentences_data:
        key = (int(e["layer"]), e["sae_lens_release"], e["sae_lens_id"])
        groups.setdefault(key, []).append(e)

    # Process groups, respecting requested layers
    for (layer, release, sae_id), entries in groups.items():
        if layer not in layer_set:
            continue

        if args.verbose:
            print(f"[LOAD] SAE layer={layer} release={release} id={sae_id}")

        sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=args.device)[0]

        for e in entries:
            h_row = int(e["index"])
            sentences = e["activating_sentences"]
            neutral = e["neutral_sentences"]
            concept = e["concept"]

            # Decoder direction for this feature
            concept_vec = sae.W_dec[h_row, :]

            scores = evaluator.evaluate_tensor(sentences, layer, concept_vec)
            random_scores = evaluator.evaluate_tensor(neutral, layer, concept_vec)

            writer.add_row(
                scores=scores,
                concept=concept,
                layer=layer,
                h_row=h_row,
                random_scores=random_scores,
            )

        # free memory
        del sae
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not args.auto_write:
        writer.write()

    if args.verbose:
        print(f"[DONE] wrote: {out_path}")


if __name__ == "__main__":
    main()
