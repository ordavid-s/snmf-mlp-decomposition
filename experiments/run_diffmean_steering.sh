PYTHONPATH=. python experiments/causal/generate_diffmean_factors.py \
  --model gpt2-small \
  --concept-data experiments/artifacts/generated_sentences.json \
  --concept-dir experiments/artifacts/diffmean_models \
  --mode mlp_out \
  --skip-existing


PYTHONPATH=. python experiments/causal/generate_diffmean_causal_output.py \
  --model-name "gpt2-small" \
  --mode "mlp_out" \
  --base-path "." \
  --vectors-dir-tpl "experiments/artifacts/diffmean_models/" \
  --save-path-tpl "experiments/artifacts/diffmean_causal_output.json" \
  --data-path experiments/artifacts/generated_sentences.json \
  --layers 0 \
  --target-kls "0.025,0.05,0.1,0.15,0.25,0.35,0.5" \
  --num-top 50 \
  --num-sent 8 \
  --base-prompt "I think that" \
  --seed 42 \
  --device auto \
  --gen-max-new-tokens 50 \
  --gen-top-k 30 \
  --gen-top-p 0.3

PYTHONPATH=. python experiments/causal/vocab_proj_diffmean.py \
  --mode mlp_out \
  --model-name gpt2-small \
  --vectors-dir experiments/artifacts/diffmean_models/ \
  --data-path   experiments/artifacts/generated_sentences.json \
  --save-path   experiments/artifacts/diffmean_vocab_proj.json \
  --layers 0 \
  --top-k 50 \
  --seed 42 \
  --device auto

PYTHONPATH=. python experiments/snmf_interp/generate_output_centric_descriptions.py\
  --input experiments/artifacts/diffmean_vocab_proj.json \
  --output experiments/artifacts/diffmean_output_descriptions.json \
  --model gpt-4o-mini \
  --layers 0 \
  --ranks 50 \
  --top-m 25 \
  --concurrency 50 \
  --max-tokens 5000

PYTHONPATH=. python experiments/causal/output_score_llm_judge.py \
  --input experiments/artifacts/diffmean_causal_output.json \
  --concepts experiments/artifacts/diffmean_output_descriptions.json \
  --output experiments/artifacts/diffmean_results_causal_out.json \
  --layers 0 \
  --ranks 50 \
  --model gpt-4o-mini \
  --concurrency 50 \
  --attempts 2 \
  --sparsity 0.01

PYTHONPATH=. python experiments/causal/input_score_llm_judge.py \
  --input experiments/artifacts/diffmean_causal_output.json \
  --concepts experiments/artifacts/input_descriptions.json \
  --output experiments/artifacts/diffmean_causal_results_in.json \
  --model gpt-4o-mini \
  --ranks 50 \
  --layers 0 \
  --concurrency 50 \
  --diffmean
