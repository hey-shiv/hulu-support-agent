"""Cluster Hulu customer messages to discover candidate intents from the data."""

import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_prep import ROOT

BRAND = "hulu_support"
K = 10  # number of clusters to start with -- we'll merge/rename by hand after

HANDLE = re.compile(r"@\w+")
URL = re.compile(r"https?://\S+")


def clean(text: pd.Series) -> pd.Series:
    text = text.str.replace(HANDLE, " ", regex=True)
    text = text.str.replace(URL, " ", regex=True)
    text = text.str.replace(r"\s+", " ", regex=True).str.strip()
    return text


def main() -> None:
    pairs = pd.read_parquet(ROOT / "data" / "processed" / "pairs.parquet")
    hulu = pairs[pairs["brand"] == BRAND].copy()
    # Only conversation openers: a mid-thread reply ("yes, during ads") has its
    # meaning in a previous turn, so it is neither classifiable in isolation nor
    # representative of what an agent receives as an incoming message.
    hulu = hulu[hulu["is_thread_start"]]
    hulu["msg_clean"] = clean(hulu["customer_msg"])
    hulu = hulu[hulu["msg_clean"].str.len() >= 15]  # drop near-empty messages

    print(f"clustering {len(hulu):,} hulu messages into {K} groups...\n")

    vec = TfidfVectorizer(max_features=3000, stop_words="english",
                          ngram_range=(1, 2), min_df=5)
    X = vec.fit_transform(hulu["msg_clean"])
    km = KMeans(n_clusters=K, random_state=0, n_init=10).fit(X)
    hulu["cluster"] = km.labels_

    terms = vec.get_feature_names_out()
    order = km.cluster_centers_.argsort()[:, ::-1]

    for c in range(K):
        sub = hulu[hulu["cluster"] == c]
        top_terms = ", ".join(terms[i] for i in order[c, :8])
        print(f"CLUSTER {c}  (n={len(sub)}, {len(sub)/len(hulu)*100:.0f}%)")
        print(f"  top words: {top_terms}")
        for m in sub["msg_clean"].sample(min(3, len(sub)), random_state=1):
            print(f"    - {m[:120]}")
        print()

    hulu.to_parquet(ROOT / "data" / "processed" / "hulu_clustered.parquet", index=False)


if __name__ == "__main__":
    main()
