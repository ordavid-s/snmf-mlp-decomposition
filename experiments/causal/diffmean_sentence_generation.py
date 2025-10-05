import asyncio
import json
import re
import time
import random
import os
from itertools import product
import argparse

from dotenv import load_dotenv
from google import genai
from tenacity import retry, wait_random_exponential, stop_after_attempt

# ── Global concurrency cap ───────────────────────────────────────────────
semaphore = asyncio.Semaphore(10)

# ── Axes for structured diversity ───────────────────────────────────────
TONES        = ["formal", "casual", "sarcastic"]
PERSPECTIVES = ["first-person", "third-person", "question"]
DOMAINS      = ["science", "politics", "nature", "conversations"]
FORMATS      = ["sentence"]
TENSES       = ["present", "past"]
SENTIMENTS   = ["positive", "negative", "neutral"]

THEMED_PROMPT = """You are given a description of an LLM concept.
Generate a grammatically correct {format} in a {tone} tone, using {perspective} perspective, about {domain}, 
in the {tense} tense with a {sentiment} sentiment, that exemplifies the concept described below.
 
Concept description:
{description}

Make sure your generated text is distinct from:
{previous_sentences}

Output only the {format} without any additional text.
"""

def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_data(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
async def generate_one_sentence(client, model, description, previous_sentences,
                                tone, perspective, domain, fmt,
                                tense, sentiment):
    prompt = THEMED_PROMPT.format(
        format=fmt,
        tone=tone,
        perspective=perspective,
        domain=domain,
        tense=tense,
        sentiment=sentiment,
        description=description,
        previous_sentences="\n".join(previous_sentences)
    )

    async with semaphore:
        resp = await client.aio.models.generate_content(
            model=model,
            contents=[prompt],
            config={"max_output_tokens": 100, "temperature": 1.0}
        )
    return resp.text.strip()

async def generate_sentences_for_combo(client, model, description,
                                       tone, perspective, domain, fmt, tense, sentiment,
                                       n=10):
    sentences = []
    attempts = 0
    while len(sentences) < n and attempts < n * 20:
        try:
            s = await generate_one_sentence(
                client, model, description, sentences,
                tone, perspective, domain, fmt, tense, sentiment
            )
            if s and s not in sentences:
                sentences.append(s)
                print(f"  ✓ [{tone}/{perspective}/{domain}/{fmt}] {len(sentences)}/{n}")
            else:
                attempts += 1
        except Exception as e:
            print(f"    ✗ retrying due to {e}")
            attempts += 1
    return sentences

async def process_all(input_json: str, output_json: str, model: str):
    # Load API key from .env
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GOOGLE_API_KEY. Create a .env file with:\nGOOGLE_API_KEY=your_key_here"
        )

    client = genai.Client(api_key=api_key)
    data   = load_data(input_json)

    # Load existing results or start fresh
    if os.path.exists(output_json):
        completed = load_data(output_json)
    else:
        completed = []

    seen = {
        (c["concept"], c["tone"], c["perspective"], c["domain"], c["format"],  c["tense"], c["sentiment"])
        for c in completed
    }

    tasks = []
    for entry in data:
        desc = entry["description"]
        for tone, persp, dom, fmt, tense, sentiment in product(
            TONES, PERSPECTIVES, DOMAINS, FORMATS, TENSES, SENTIMENTS
        ):
            key = (desc, tone, persp, dom, fmt, tense, sentiment)
            if key in seen:
                continue

            async def task_fn(desc=desc, tone=tone, persp=persp, dom=dom, fmt=fmt, tense=tense, sentiment=sentiment):
                print(f"Generating for concept '{desc[:30]}…', {tone}/{persp}/{dom}/{fmt}")
                sents = await generate_sentences_for_combo(
                    client, model, desc, tone, persp, dom, fmt, tense, sentiment, n=10
                )
                rec = {
                    "concept":     desc,
                    "tone":        tone,
                    "perspective": persp,
                    "domain":      dom,
                    "format":      fmt,
                    "tense":       tense,
                    "sentiment":   sentiment,
                    "sentences":   sents
                }
                completed.append(rec)
                # Save incrementally
                save_data(output_json, completed)

            tasks.append(task_fn())

    print(f"Scheduling {len(tasks)} generation tasks…")
    for t in asyncio.as_completed(tasks):
        await t

    print("All done.")

def parse_args():
    p = argparse.ArgumentParser(description="Generate themed sentences for concept descriptions.")
    p.add_argument("--input", required=True, help="Path to input JSON of concept descriptions.")
    p.add_argument("--output", required=True, help="Path to write generated sentences JSON.")
    p.add_argument("--model", default="gemini-2.0-flash", help="Model name (default: gemini-2.0-flash).")
    return p.parse_args()

def main():
    args = parse_args()
    asyncio.run(process_all(args.input, args.output, args.model))

if __name__ == "__main__":
    main()
