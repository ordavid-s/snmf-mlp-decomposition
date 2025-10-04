PYTHONPATH=. python experiments/concept_detection/generate_sentences.py \
  --input-json data/gpt2_mlp_features.json \
  --output-json experiments/artifacts/sae_generated_sentences.json \
  --model gpt-4o-mini \
  --layers 0 \
  --n-per-mode 5 \
  --concurrency 50 \
  --max-tokens 100 \
  --retries 3 \
  --jitter-min-ms 50 \
  --jitter-max-ms 300 \
  --env-var OPENAI_API_KEY


PYTHONPATH=. python experiments/concept_detection/benchmark_sae.py \
  --model-name gpt2-small \
  --layers 0 \
  --hook-template "blocks.{layer_number}.hook_mlp_out" \
  --sentences-json experiments/artifacts/sae_generated_sentences.json \
  --concept-json data/gpt2_mlp_features.json \
  --save-path experiments/artifacts/sae_interpretability_results.json \
  --device mps \
  --overwrite \
  --verbose
