"""Generate SILVER labels: LLM-assigned intents on non-golden messages.

Silver labels are explicitly NOT ground truth. They are never used to score
anything. They exist for one purpose: to give the TF-IDF baseline something to
train on, so that scarce human labels can all stay in the test set.

The circularity is real and stated plainly: a TF-IDF model trained on this
LLM's labels is distilling that LLM, so it cannot meaningfully exceed it. That
makes it useful for one specific question -- can a linear model that runs in
microseconds recover most of the LLM's behaviour? -- and useless as an
independent check. The keyword-rules baseline is the independent one.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.agent import classify
from src.data_prep import ROOT

N_SILVER = 400
SEED = 5


def main() -> None:
    pool = pd.read_parquet(ROOT / "data" / "processed" / "hulu_clustered.parquet")
    gold = pd.read_csv(ROOT / "data" / "golden" / "golden_labelled.csv")

    pool = pool[~pool["customer_msg"].isin(set(gold["customer_msg"]))]
    if "thread_root" in pool and "thread_root" in gold:
        pool = pool[~pool["thread_root"].isin(set(gold["thread_root"].dropna()))]
    pool = pool.drop_duplicates("msg_clean")

    sample = pool.sample(min(N_SILVER, len(pool)), random_state=SEED).reset_index(drop=True)
    print(f"labelling {len(sample)} non-golden messages with the LLM (silver)...")

    labels = []
    for i, msg in enumerate(sample["customer_msg"], 1):
        labels.append(classify(msg)["intent"])
        if i % 50 == 0:
            print(f"  {i}/{len(sample)}", flush=True)

    sample["intent"] = labels
    out = ROOT / "data" / "processed" / "silver_labels.parquet"
    sample[["customer_msg", "msg_clean", "intent"]].to_parquet(out, index=False)
    print(f"\nsaved -> {out}")
    print(sample["intent"].value_counts().to_string())


if __name__ == "__main__":
    main()
