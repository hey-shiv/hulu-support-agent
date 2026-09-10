"""Plot the Hulu message clusters in 2D so we can see what KMeans actually found."""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_prep import ROOT

BRAND = "hulu_support"
K = 10

HANDLE = re.compile(r"@\w+")
URL = re.compile(r"https?://\S+")


def clean(text: pd.Series) -> pd.Series:
    text = text.str.replace(HANDLE, " ", regex=True)
    text = text.str.replace(URL, " ", regex=True)
    return text.str.replace(r"\s+", " ", regex=True).str.strip()


def main() -> None:
    pairs = pd.read_parquet(ROOT / "data" / "processed" / "pairs.parquet")
    hulu = pairs[pairs["brand"] == BRAND].copy()
    hulu["msg_clean"] = clean(hulu["customer_msg"])
    hulu = hulu[hulu["msg_clean"].str.len() >= 15]

    vec = TfidfVectorizer(max_features=3000, stop_words="english",
                          ngram_range=(1, 2), min_df=5)
    X = vec.fit_transform(hulu["msg_clean"])
    labels = KMeans(n_clusters=K, random_state=0, n_init=10).fit_predict(X)

    # Compress 3000 TF-IDF dimensions down to 2, for plotting only.
    # This does NOT change the clustering -- KMeans already ran on the full
    # 3000-dim data above. This is purely so we can look at the result.
    coords = TruncatedSVD(n_components=2, random_state=0).fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10",
                         s=6, alpha=0.5)
    ax.set_title(f"Hulu support messages, {K} TF-IDF/KMeans clusters "
                f"(n={len(hulu):,})")
    ax.set_xlabel("SVD component 1")
    ax.set_ylabel("SVD component 2")
    legend = ax.legend(*scatter.legend_elements(), title="cluster", loc="best")
    ax.add_artist(legend)

    out = ROOT / "reports" / "figures" / "hulu_clusters.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
