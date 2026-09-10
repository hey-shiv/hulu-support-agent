"""Find historical Hulu exchanges most similar to a new customer message."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_prep import ROOT
from src.embeddings import embed

PROC = ROOT / "data" / "processed"


def load_index() -> tuple[np.ndarray, pd.DataFrame]:
    vectors = np.load(PROC / "hulu_index_vectors.npy")
    pairs = pd.read_parquet(PROC / "hulu_index_pairs.parquet")
    return vectors, pairs


def retrieve(query: str, vectors: np.ndarray, pairs: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Return the k historical pairs whose customer message is most similar to query."""
    q = embed(query)

    # Cosine similarity of q against every row of `vectors` at once, instead of
    # one row at a time in a loop. `vectors @ q` is a matrix-vector product:
    # for each row, it computes the dot product with q, all in one call.
    dots = vectors @ q
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(q)
    sims = dots / norms

    top_k_idx = np.argsort(sims)[::-1][:k]  # highest similarity first

    result = pairs.iloc[top_k_idx].copy()
    result["similarity"] = sims[top_k_idx]
    return result
