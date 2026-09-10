"""Turn text into embedding vectors, with a disk cache so we never re-embed
the same message twice. Same idea as llm.py's cache: hash the input, use that
as the filename, skip the model call entirely if the file already exists.
"""

import hashlib
from pathlib import Path

import numpy as np
import ollama

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache" / "embeddings"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "nomic-embed-text"


def _cache_path(text: str) -> Path:
    key = hashlib.sha256(text.encode()).hexdigest()[:24]
    return CACHE_DIR / f"{key}.npy"


def embed(text: str) -> np.ndarray:
    """Embed one piece of text, reusing a cached vector if we've seen it before."""
    path = _cache_path(text)
    if path.exists():
        return np.load(path)
    vec = np.array(ollama.embed(model=MODEL, input=text)["embeddings"][0])
    np.save(path, vec)
    return vec


def embed_many(texts: list[str], batch_size: int = 200) -> np.ndarray:
    """Embed many texts at once.

    Calling the model one text at a time costs ~0.55s per call, almost all of
    it fixed overhead -- measured on this machine, batching 100 texts into one
    call brought that to ~6ms per text, roughly 90x faster. So: check the cache
    for each text first, then send only the *uncached* ones to the model in
    batches, and cache each result individually as it comes back.
    """
    results: list[np.ndarray | None] = [None] * len(texts)
    missing_idx, missing_text = [], []

    for i, t in enumerate(texts):
        path = _cache_path(t)
        if path.exists():
            results[i] = np.load(path)
        else:
            missing_idx.append(i)
            missing_text.append(t)

    for start in range(0, len(missing_text), batch_size):
        batch = missing_text[start : start + batch_size]
        idxs = missing_idx[start : start + batch_size]
        response = ollama.embed(model=MODEL, input=batch)
        for idx, text, vec in zip(idxs, batch, response["embeddings"]):
            vec = np.array(vec)
            np.save(_cache_path(text), vec)
            results[idx] = vec

    return np.stack(results)
