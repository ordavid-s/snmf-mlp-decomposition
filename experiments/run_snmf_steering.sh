PYTHONPATH=. python experiments/train/train.py \
    --sparsity 0.01 \
    --ranks 50 \
    --max-iterations-per-layer 2000 \
    --patience 1500 \
    --model-name gpt2-small \
    --factorization-mode mlp \
    --layers 0 \
    --data-path data/final_dataset_20_concepts.json \
    --model-device mps \
    --data-device cpu \
    --fitting-device mps \
    --base-path . \
    --save-path experiments/artifacts/ \
    --seed 42

PYTHONPATH=. python experiments/snmf_interp/generate_concept_context.py \
  --models-dir experiments/artifacts \
  --output-json experiments/artifacts/concept_contexts.json \
  --layers 0 \
  --ranks 50 \
  --num-samples-per-factor 25 \
  --context-window 15 \
  --sparsity 0.01 \
  --seed 42 \
  --model-name "gpt2-small" \
  --factor-mode mlp \
  --data-path data/final_dataset_20_concepts.json \
  --model-device mps \
  --data-device cpu


PYTHONPATH=. python experiments/snmf_interp/generate_input_descriptions.py \
  --input-json experiments/artifacts/concept_contexts.json \
  --output-json experiments/artifacts/input_descriptions.json \
  --model gpt-4o-mini \
  --env-var OPENAI_API_KEY \
  --layers 0 \
  --k-values 50 \
  --top-m 10 \
  --max-tokens 200 \
  --concurrency 50 \
  --retries 5

PYTHONPATH=. python experiments/causal/generate_vocab_proj.py\
  --model-name gpt2-small \
  --base-path . \
  --factorization-base-path experiments/artifacts \
  --output-path experiments/artifacts/vocab_proj.json \
  --layers 0 \
  --ranks 50 \
  --top-k 75 \
  --sparsity 0.01 \
  --device mps \
  --seed 123

PYTHONPATH=. python experiments/snmf_interp/generate_output_centric_descriptions.py\
  --input experiments/artifacts/vocab_proj.json \
  --output experiments/artifacts/output_descriptions.json \
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

PYTHONPATH=. python experiments/causal/input_score_llm_judge.py \
  --input experiments/artifacts/causal_output.json \
  --concepts experiments/artifacts/input_descriptions.json \
  --output experiments/artifacts/causal_results_in.json \
  --model gpt-4o-mini \
  --ranks 50 \
  --layers 0 \
  --concurrency 50

PYTHONPATH=. python experiments/causal/output_score_llm_judge.py \
  --input experiments/artifacts/causal_output.json \
  --concepts experiments/artifacts/output_descriptions.json \
  --output experiments/artifacts/results_causal_out.json \
  --layers 0 \
  --ranks 50 \
  --model gpt-4o-mini \
  --concurrency 50 \
  --attempts 2 \
  --sparsity 0.01
