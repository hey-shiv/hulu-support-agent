"""Call a local LLM (via Ollama) for text generation, with a disk cache.

Same idea as embeddings.py: hash the exact prompt, skip the model call if we've
already made this exact request. This is what makes `make reproduce` fast --
every prompt we ran during development is replayable with zero model calls.
"""

import hashlib
import json
from pathlib import Path

import ollama

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache" / "llm"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(model: str, system: str, prompt: str) -> Path:
    key = hashlib.sha256(f"{model}|{system}|{prompt}".encode()).hexdigest()[:24]
    return CACHE_DIR / f"{key}.txt"


def generate(prompt: str, model: str, system: str = "") -> str:
    path = _cache_path(model, system, prompt)
    if path.exists():
        return path.read_text()

    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    response = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": 0.0, "seed": 0},  # deterministic, for reproducibility
    )
    text = response["message"]["content"]
    path.write_text(text)
    return text


def generate_json(prompt: str, model: str, system: str = "") -> dict:
    """Same as generate(), but pulls out the first JSON object in the response.

    Small local models often wrap JSON in prose or markdown fences even when
    told not to, so we search for it rather than trust the raw string. Returns
    {} on unparseable output -- that failure gets counted and reported, not hidden.
    """
    return _extract_json(generate(prompt, model, system=system))


def _extract_json(raw: str) -> dict:
    """Pull the first complete JSON object out of a model response.

    Tracks brace depth rather than taking everything between the first '{' and
    the last '}', because trailing prose containing braces would otherwise be
    swallowed into the parse. Split out from generate_json so it is testable
    without calling a model.
    """
    start = raw.find("{")
    if start == -1:
        return {}
    depth = 0
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(raw[start : i + 1])
                except json.JSONDecodeError:
                    return {}
                return parsed if isinstance(parsed, dict) else {}
    return {}
