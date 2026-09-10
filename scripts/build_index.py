"""Build the retrieval index from the REFERENCE half of the corpus only.

The split that matters for trusting any result in this repo:

  evaluation set - the hand-labelled golden examples. Never in the index.
  reference set  - everything else. This is all retrieval may ever see.

An earlier version indexed the entire Hulu corpus, golden examples included.
Verified consequence: querying with a test message returned that same message
at similarity 1.0000 as its own top "similar past exchange", handing the agent
the real historical answer as grounding evidence and letting the copy-nearest
reply baseline emit the literal ground truth. Every reply-quality number from
that version measured memorisation.

We exclude by conversation thread, not just by exact message, because two
messages from the same thread describe the same incident -- retrieving a
sibling turn would leak the same answer through a different row.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_prep import ROOT
from src.embeddings import embed_many

BRAND = "hulu_support"
PROC = ROOT / "data" / "processed"


def main() -> None:
    pairs = pd.read_parquet(PROC / "pairs.parquet")
    hulu = pairs[pairs["brand"] == BRAND].reset_index(drop=True)
    print(f"hulu corpus: {len(hulu):,} exchanges")

    golden_path = ROOT / "data" / "golden" / "golden_labelled.csv"
    if golden_path.exists():
        golden = pd.read_csv(golden_path)
        held_out_msgs = set(golden["customer_msg"])
        held_out_threads = set(golden["thread_root"].dropna()) if "thread_root" in golden else set()

        before = len(hulu)
        mask = (hulu["customer_msg"].isin(held_out_msgs)
                | hulu["thread_root"].isin(held_out_threads))
        hulu = hulu[~mask].reset_index(drop=True)
        print(f"excluded {before - len(hulu)} rows belonging to golden examples "
              f"or their conversation threads")
    else:
        print("WARNING: no golden set found -- index will contain everything")

    print(f"reference set for retrieval: {len(hulu):,} exchanges")
    vectors = embed_many(hulu["customer_msg"].tolist())

    np.save(PROC / "hulu_index_vectors.npy", vectors)
    hulu.to_parquet(PROC / "hulu_index_pairs.parquet", index=False)
    print(f"saved {vectors.shape[0]} vectors, dim={vectors.shape[1]}")


if __name__ == "__main__":
    main()
