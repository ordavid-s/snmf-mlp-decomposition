import asyncio
import json
import os
import re
import argparse
from typing import List, Any
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv

# ---------------- Prompts ---------------- #
CONCEPT_PROMPT = """
You are given a set of tokens, their surrounding context (words before and after the token), and an importance score.
Your task is to determine what is the connection between all the tokens.

### Instructions:

1. **Focus on High-Importance Samples:**  
   Examine only the token-context pairs with the highest importance scores. If a significant drop is observed beyond a threshold, ignore the lower-scoring pairs.
   
2. **Assess Token Consistency vs. Contextual Patterns:**  
   - **Token Consistency:** If the high-importance samples are mostly identical tokens or strongly related tokens, then consider the tokens only as the primary contributor.  
   - **Contextual Patterns:** If the tokens are not related to one another, then focus on common semantic, syntactic, or structural patterns in the surrounding contexts.

3. Always choose the simplest and most obvious underlying connection. If inspecting the tokens alone is enough to find a connection, do not mention or utilize the contexts.
   
### Output Format:

Analysis:
<reason what the underlying connection is.>

Results:
<Single sentence description of the single most obvious connection between the tokens>

### Input:
 
Token-Context Pairs:
```{token_context_str}```

Remember, find the most obvious connection prioritizing connecting the tokens alone without the contexts and only if you cannot find any connection between the tokens then you may inspect their contexts.
"""

# ---------------- Helpers ---------------- #
def extract_results_section(text: str) -> str | None:
    m = re.search(r"Results:\s*(.*)", text, re.DOTALL)
    return m.group(1).strip() if m else None

def load_data(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def save_data(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def _parse_int_list(csv: str) -> List[int]:
    if not csv:
        return []
    return [int(x.strip()) for x in csv.split(",") if x.strip()]

def _to_float_activation(x) -> float:
    """Handle values like 0.123, '0.123', 'tensor(0.123)' safely."""
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    s = s.replace("tensor(", "").replace(")", "")
    try:
        return float(s)
    except ValueError:
        return float("nan")

# ---------------- Argparse ---------------- #
def build_arg_parser():
    p = argparse.ArgumentParser(description="Generate concept descriptions from token-context activations.")
    # I/O
    p.add_argument("--input-json", default="rebuttal/init_methods/svd/concept_contexts.json",
                   help="Path to input JSON containing top_activations per (K, layer, h_row).")
    p.add_argument("--output-json", default="rebuttal/init_methods/svd/input_descriptions.json",
                   help="Path to write the output descriptions JSON.")
    # OpenAI
    p.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name.")
    p.add_argument("--env-var", default="OPENAI_API_KEY",
                   help="Environment variable holding the API key (loaded via python-dotenv if present).")
    # Filtering / selection
    p.add_argument("--layers", default="0,6,12,18,25,31",
                   help="Comma-separated list of layer indices to include.")
    p.add_argument("--k-values", default="100",
                   help="Comma-separated list of K (rank) values to include.")
    # Generation controls
    p.add_argument("--top-m", type=int, default=10, help="Number of top activations to consider.")
    p.add_argument("--max-tokens", type=int, default=200, help="max_tokens for each completion.")
    p.add_argument("--concurrency", type=int, default=50, help="Semaphore limit for concurrent calls.")
    p.add_argument("--retries", type=int, default=5, help="Tenacity stop_after_attempt.")
    return p

# ---------------- Async workers ---------------- #
def make_generate_concept(retries: int, model: str, max_tokens: int, semaphore: asyncio.Semaphore):
    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(retries))
    async def _inner(client: AsyncOpenAI, entry, top_m: int):
        # sort and pick top-M by activation
        top = sorted(
            entry["top_activations"],
            key=lambda x: _to_float_activation(x["activation"]),
            reverse=True
        )[:top_m]

        token_context_str = "\n".join(
            f"Token: `{act['token']}`, Context: `{act['context']}` | Score: `{_to_float_activation(act['activation'])}`"
            for act in top
        )

        async with semaphore:
            resp: ChatCompletion = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": CONCEPT_PROMPT.format(token_context_str=token_context_str)}],
                max_tokens=max_tokens,
            )
        return extract_results_section(resp.choices[0].message.content.strip())
    return _inner

def make_process_entry(generate_concept):
    async def _inner(client: AsyncOpenAI, entry, top_m: int):
        concept_desc = await generate_concept(client, entry, top_m)
        print(".", end="", flush=True)
        return {
            "K": entry["K"],
            "layer": entry["layer"],
            "h_row": entry["h_row"],
            "description": concept_desc,
        }
    return _inner

# ---------------- Main ---------------- #
async def run(args):
    data = load_data(args.input_json)
    layers = set(_parse_int_list(args.layers))
    k_values = set(_parse_int_list(args.k_values))

    # Load API key
    load_dotenv()
    api_key = os.getenv(args.env_var)
    if not api_key:
        raise RuntimeError(f"API key not found in environment variable '{args.env_var}'.")

    semaphore = asyncio.Semaphore(args.concurrency)
    client = AsyncOpenAI(api_key=api_key)

    generate_concept = make_generate_concept(
        retries=args.retries, model=args.model, max_tokens=args.max_tokens, semaphore=semaphore
    )
    process_entry = make_process_entry(generate_concept)

    tasks = [
        process_entry(client, e, args.top_m)
        for e in data
        if (int(e["layer"]) in layers) and (int(e["K"]) in k_values)
    ]

    print(f"Running over {len(tasks)} tasks …")
    results = []
    for coro in asyncio.as_completed(tasks):
        results.append(await coro)

    save_data(args.output_json, results)
    print(f"\nWrote {len(results)} descriptions to {args.output_json}")

def main():
    args = build_arg_parser().parse_args()
    asyncio.run(run(args))

if __name__ == "__main__":
    main()
