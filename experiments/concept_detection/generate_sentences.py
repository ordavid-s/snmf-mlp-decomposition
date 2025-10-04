import asyncio
import json
import re
import os
import random
import argparse
from typing import List
from openai import AsyncOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv

# ---------------- Prompts (kept as constants; override in code if needed) ---------------- #
ACTIVATING_GENERATION_PROMPT = """You are given a description of an LLM concept.
Given the description, generate a sentence that contains tokens that would activate the feature. Make sure your generated sentence exemplifies all the key terms and structures specified in the description.
Make sure that the sentences are full grammatically correct english sentences.
Category description:
{description}

Make sure you generate a sentence that is distinct from:
{previous_sentences}

Output only the sentence without any additional text.
"""

NEUTRAL_GENERATION_PROMPT = """You are given a description of an LLM concept.
Your objective is to generate a neutral sentence that should not activate the feature.
This means that you must not include in the sentence any tokens that relate to the feature.
Have the generated sentence be on a completely unrelated topic.

Concept description:
{description}

Make sure you generate a sentence that is distinct from:
{previous_sentences}

Output only the sentence without any additional text.
Make sure to exclude any tokens that may activate the concept.
"""

def extract_results_section(text):
    match = re.search(r"Results:\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_data(file_path, data):
    print(data)
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def _parse_int_list(csv: str) -> List[int]:
    if not csv:
        return []
    return [int(x.strip()) for x in csv.split(",") if x.strip() != ""]


def build_arg_parser():
    p = argparse.ArgumentParser(description="Generate activating/neutral sentences for concept descriptions.")
    # Paths / model
    p.add_argument("--input-json", default="rebuttal/init_methods/svd/input_descriptions.json",
                   help="Path to input JSON with concept entries.")
    p.add_argument("--output-json", default="rebuttal/init_methods/svd/generated_sentences_svd.json",
                   help="Path to write generated sentences JSON.")
    p.add_argument("--model", default="gpt-4o-mini",
                   help="OpenAI model name.")
    # Filtering / counts
    p.add_argument("--layers", default="0,6,12,18,25,31",
                   help="Comma-separated list of layer indices to include (e.g., '0,6,12').")
    p.add_argument("--k-values", default="100",
                   help="Comma-separated list of K values to include (e.g., '50,100').")
    p.add_argument("--n-per-mode", type=int, default=5,
                   help="Number of sentences to generate per mode (activating and neutral).")
    # Runtime knobs
    p.add_argument("--concurrency", type=int, default=50,
                   help="Max concurrent API calls (semaphore).")
    p.add_argument("--max-tokens", type=int, default=100,
                   help="max_tokens for each completion.")
    p.add_argument("--retries", type=int, default=3,
                   help="Tenacity: stop_after_attempt.")
    p.add_argument("--jitter-min-ms", type=int, default=50,
                   help="Minimum sleep jitter (ms) before starting an entry.")
    p.add_argument("--jitter-max-ms", type=int, default=300,
                   help="Maximum sleep jitter (ms) before starting an entry.")
    # API Key/env
    p.add_argument("--env-var", default="OPENAI_API_KEY",
                   help="Environment variable name holding the API key.")
    return p

def make_generate_one_sentence(retries: int, model: str, max_tokens: int, semaphore: asyncio.Semaphore):
    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(retries))
    async def _inner(client, description, previous_sentences, mode):
        if mode == "activating":
            prompt = ACTIVATING_GENERATION_PROMPT.format(
                description=description,
                previous_sentences="\n".join(previous_sentences)
            )
        elif mode == "neutral":
            prompt = NEUTRAL_GENERATION_PROMPT.format(
                description=description,
                previous_sentences="\n".join(previous_sentences)
            )
        else:
            raise ValueError("Mode must be either 'activating' or 'neutral'.")

        async with semaphore:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
        return response.choices[0].message.content.strip()
    return _inner

def make_generate_sentences(generate_one):
    async def _inner(client, concept_response, n, mode):
        sentences = []
        retries = 0
        while len(sentences) < n and retries < n * 2:
            sentence = await generate_one(client, concept_response, sentences, mode)
            if sentence and sentence not in sentences:
                sentences.append(sentence)
                print(f"Generated {mode} sentence {len(sentences)}/{n}")
            else:
                retries += 1
        return sentences
    return _inner

def make_process_entry(generate_sentences, jitter_min_ms: int, jitter_max_ms: int):
    async def _inner(client, entry, n_per_mode: int):
        print(f"Starting processing for entry K: {entry.get('K', 'SAE no K')} Layer: {entry['layer']} Idx: {entry.get('h_row', entry.get('index', None))}")
        # Spread out concept generation to avoid bursty calls
        await asyncio.sleep(random.uniform(jitter_min_ms / 1000.0, jitter_max_ms / 1000.0))

        activating_sentences = await generate_sentences(client, entry['description'], n=n_per_mode, mode="activating")
        neutral_sentences    = await generate_sentences(client, entry['description'], n=n_per_mode, mode="neutral")

        print(f"Completed processing for entry K:  {entry.get('K', 'SAE no K')} Layer: {entry['layer']} Idx: {entry.get('h_row', entry.get('index', None))}")
        return {
            "K": entry.get('K', 'SAE'),
            "h_row": entry.get('h_row', entry.get('index', None)),
            "layer": entry['layer'],
            "activating_sentences": activating_sentences,
            "neutral_sentences": neutral_sentences,
            "concept": entry['description'],
        }
    return _inner

async def process_all_data(
    input_json: str,
    output_json: str,
    model: str,
    layers_csv: str,
    k_values_csv: str,
    n_per_mode: int,
    concurrency: int,
    max_tokens: int,
    retries: int,
    env_var: str,
    jitter_min_ms: int,
    jitter_max_ms: int,
):
    data = load_data(input_json)
    layers = set(_parse_int_list(layers_csv))
    k_values = set(_parse_int_list(k_values_csv))

    # Load API key from .env / env var
    load_dotenv()
    api_key = os.getenv(env_var)
    if not api_key:
        raise RuntimeError(f"API key not found in environment variable '{env_var}'.")

    semaphore = asyncio.Semaphore(concurrency)
    client = AsyncOpenAI(api_key=api_key)

    generate_one_sentence = make_generate_one_sentence(retries=retries, model=model, max_tokens=max_tokens, semaphore=semaphore)
    generate_sentences = make_generate_sentences(generate_one_sentence)
    process_entry = make_process_entry(generate_sentences, jitter_min_ms=jitter_min_ms, jitter_max_ms=jitter_max_ms)

    tasks = [
        process_entry(client, entry, n_per_mode)
        for entry in data
        if (entry.get('K', True) or int(entry['K']) in k_values) and (int(entry['layer']) in layers)
    ]
    total_tasks = len(tasks)
    print(f"Processing {total_tasks} items concurrently... (model={model}, n_per_mode={n_per_mode}, max_tokens={max_tokens})")

    results = []
    completed_tasks = 0
    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)
        completed_tasks += 1
        print(f"Completed {completed_tasks}/{total_tasks} tasks")

    save_data(output_json, results)
    print("Processing complete. Results written to", output_json)

def main():
    args = build_arg_parser().parse_args()
    asyncio.run(
        process_all_data(
            input_json=args.input_json,
            output_json=args.output_json,
            model=args.model,
            layers_csv=args.layers,
            k_values_csv=args.k_values,
            n_per_mode=args.n_per_mode,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            retries=args.retries,
            env_var=args.env_var,
            jitter_min_ms=args.jitter_min_ms,
            jitter_max_ms=args.jitter_max_ms,
        )
    )

if __name__ == '__main__':
    main()
