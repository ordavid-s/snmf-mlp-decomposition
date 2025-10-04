import os
import sys
import json
import re
import asyncio
import argparse
from typing import List, Optional, Tuple

# If your script is in ./scripts/, this keeps project imports working:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from experiments.evaluation.json_handler import JsonHandler

# ── utils ─────────────────────────────────────────────────────────────────────
def parse_int_list(spec: Optional[str]) -> Optional[List[int]]:
    """
    Parse '0,1,2' or '0-3' or '0,2,5-7' into a list of ints.
    Returns None if spec is None or empty.
    """
    if not spec:
        return None
    out: List[int] = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            a, b = int(a), int(b)
            if a <= b:
                out.extend(range(a, b + 1))
            else:
                out.extend(range(a, b - 1, -1))
        else:
            out.append(int(part))
    return out

def extract_results_section(text: str) -> Optional[str]:
    m = re.search(r"Results:\s*(.*)", text, re.DOTALL)
    return m.group(1).strip() if m else None

def load_data(path: str) -> list[dict]:
    with open(path, 'r') as f:
        return json.load(f)

def save_json(path: str, data) -> None:
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

# ── core ──────────────────────────────────────────────────────────────────────
async def run(args):
    # .env
    load_dotenv()  # discovers OPENAI_API_KEY and optional defaults

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment (.env).")

    # env-provided defaults (can be overridden by CLI)
    env_input  = os.getenv("INPUT_JSON")
    env_output = os.getenv("OUTPUT_JSON")
    env_model  = os.getenv("MODEL")
    env_top_m  = os.getenv("TOP_M")
    env_conc   = os.getenv("CONCURRENCY")

    input_json  = args.input or env_input  or 'rebuttal/init_methods/svd/vocab_proj_svd.json'
    output_json = args.output or env_output or 'rebuttal/init_methods/svd/output_descriptions_svd.json'
    model       = args.model  or env_model  or 'gpt-4o-mini'
    top_m       = args.top_m  if args.top_m is not None else (int(env_top_m) if env_top_m else 25)
    concurrency = args.concurrency if args.concurrency is not None else (int(env_conc) if env_conc else 50)
    max_tokens  = args.max_tokens

    layers = parse_int_list(args.layers) or [23, 31]
    ranks  = parse_int_list(args.ranks)  or []

    # client & semaphore
    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)

    CONNECTION_PROMPT = """
You are given a set of tokens and their importance score.
Your task is to determine what is the connection between all the tokens.

### Instructions:

1. **Focus on High-Importance Samples:**  
   Examine only the token pairs-score with the highest importance scores. If a significant drop is observed beyond a threshold, ignore the lower-scoring tokens.

2. Always choose the simplest and most obvious underlying connection. 
   
3. Ignore noisy tokens. Some tokens may be unrelated to the concept, only consider tokens that share the most obvious connection.

4. If there is no concept or underlying connection between the tokens output under the Results section: TRASH

### Output Format:

Analysis:
<reason about what the underlying connection is.>

Results:
- Connection: <Single sentence description of the single most obvious connection between the tokens>

### Token-Score Pairs:
```{token_context_str}```

Do not add any markdown to the examples.
Make sure you output a very precise and detailed description of the concept that fully captures it.
""".strip()

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
    async def _call_openai(messages: list[dict]):
        async with semaphore:
            return await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )

    async def generate_concept(entry: dict) -> dict:
        # pair tokens & scores, sort, take top M
        pairs = list(zip(entry['top_shifted_tokens'], entry['top_logit_values']))
        top = sorted(pairs, key=lambda x: float(x[1]), reverse=True)[:top_m]
        token_context_str = "\n".join(
            f"Token: {tok[0][0]} | Importance Score: {scr}" for tok, scr in top
        )

        prompt = CONNECTION_PROMPT.format(token_context_str=token_context_str)
        print(f"[→] Generating for K={entry['K']} layer={entry['layer']} row={entry['h_row']}…", flush=True)

        resp = await _call_openai([{"role": "user", "content": prompt}])
        content = resp.choices[0].message.content
        result  = extract_results_section(content) or "ERROR: no Results section"
        print(f"[✔] Done K={entry['K']} layer={entry['layer']}", flush=True)
        return {
            'description': result,
            'layer': entry['layer'],
            'K': entry['K'],
            'h_row': entry['h_row'],
            'sign': entry.get('intervention_sign')
        }

    # filter data
    data = load_data(input_json)
    filtered = [
        e for e in data
        if int(e['layer']) in layers and (not ranks or int(e['K']) in ranks)
    ]

    print(f"Processing {len(filtered)} entries…", flush=True)
    tasks = [generate_concept(e) for e in filtered]

    results = []
    for coro in asyncio.as_completed(tasks):
        try:
            res = await coro
            results.append(res)
            print(f"  → Completed {len(results)}/{len(filtered)}", flush=True)
        except Exception as err:
            print(f"  ⚠ Sample exception: {err}", flush=True)

    json_handler = JsonHandler(
        ["description", "layer", "h_row", "K", "sign"],
        output_json,
        auto_write=False
    )
    for row in results:
        json_handler.add_row(**row)
    json_handler.write()

    bad  = sum(('TRASH' in (r['description'] or "")) for r in results)
    good = len(results) - bad
    print(f"Good: {good}\tBad: {bad}", flush=True)

# ── entrypoint ────────────────────────────────────────────────────────────────
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate concept descriptions from token-score pairs using OpenAI.")
    # IO & model
    p.add_argument("--input", "-i", type=str, help="Path to input JSON (overrides INPUT_JSON env).")
    p.add_argument("--output", "-o", type=str, help="Path to output JSON (overrides OUTPUT_JSON env).")
    p.add_argument("--model", "-m", type=str, help="Model name (overrides MODEL env).")
    # Selection / behavior
    p.add_argument("--layers", type=str, help="Layers filter like '23,31' or '0-3,6'. Default env/23,31.")
    p.add_argument("--ranks",  type=str, help="Ranks (K) filter like '50,100' or '50-200'. If omitted, no K filter.")
    p.add_argument("--top-m", type=int, help="Top-M tokens to consider per entry (overrides TOP_M env).")
    p.add_argument("--concurrency", type=int, help="Max concurrent API calls (overrides CONCURRENCY env).")
    p.add_argument("--max-tokens", type=int, default=5000, help="max_tokens for the completion (default 5000).")
    return p

def main():
    args = build_argparser().parse_args()
    asyncio.run(run(args))

if __name__ == "__main__":
    main()
