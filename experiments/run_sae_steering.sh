PYTHONPATH=. python experiments/sae_interp/generate_vocab_proj_sae.py\
  --concept-data data/gpt2_mlp_features.json \
  --out-json     experiments/artifacts/sae_vocab_proj.json \
  --model-name   gpt2-small \
  --device       mps \
  --seed         42 \
  --top-k        50 \
  --intervention-type mlp_out \
  --only-layers  0 \
  --auto-write


PYTHONPATH=. python experiments/snmf_interp/generate_output_centric_descriptions.py\
  --input experiments/artifacts/sae_vocab_proj.json \
  --output experiments/artifacts/sae_output_descriptions.json \
  --model gpt-4o-mini \
  --layers 0 \
  --ranks 50 \
  --top-m 25 \
  --concurrency 50 \
  --max-tokens 5000

PYTHONPATH=. python experiments/causal/generate_causal_output.py \
  --model-name gpt2-small \
  --layers 0 \
  --ranks 50 \
  --sparsity 0.01 \
  --factorization-base-path experiments/artifacts \
  --save-path experiments/artifacts/causal_output.json \
  --device mps

PYTHONPATH=. python experiments/causal/generate_sae_causal_output.py \
  --concept-json  data/gpt2_mlp_features.json \
  --save-json    experiments/artifacts/sae_causal_output.json \
  --model-name   gpt2-small \
  --intervention-type mlp_out \
  --base-prompt "I think that" \
  --target-kls  "0.025,0.05,0.1,0.15,0.25,0.35,0.5" \
  --num-top-logits 50 \
  --num-sentences 8 \
  --gen-max-new 50 \
  --gen-top-k 30 \
  --gen-top-p 0.3 \
  --seed 42 \
  --device mps \
  --include-layers 0


PYTHONPATH=. python experiments/causal/input_score_llm_judge.py \
  --input experiments/artifacts/sae_causal_output.json \
  --concepts data/gpt2_mlp_features.json \
  --output experiments/artifacts/sae_causal_results_in.json \
  --model gpt-4o-mini \
  --ranks 50 \
  --layers 0 \
  --concurrency 50

PYTHONPATH=. python experiments/causal/output_score_llm_judge.py \
  --input experiments/artifacts/sae_causal_output.json \
  --concepts experiments/artifacts/sae_output_descriptions.json \
  --output experiments/artifacts/sae_results_causal_out.json \
  --layers 0 \
  --ranks 50 \
  --model gpt-4o-mini \
  --concurrency 50 \
  --attempts 2 \
  --sparsity 0.01
