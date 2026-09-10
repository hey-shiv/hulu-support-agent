"""Draw the golden evaluation set (target: 200 examples, assignment asks 150-250).

Four strata, each drawn to answer a different question. Every row keeps its
stratum tag so results are never blended into one number that hides which
population it describes.

  natural              - plain random draw. The only slice that reflects real
                         traffic mix, so it is what predicts production
                         behaviour.
  rare_boost           - spread evenly across all 10 TF-IDF clusters, so
                         low-frequency topics get enough examples to have a
                         per-class metric at all.
  escalation_sensitive - messages matching billing/account/refund/fraud
                         language. These are the highest-cost errors in the
                         whole system (a wrong auto-reply about money is worse
                         than any playback mistake), and the first 60-example
                         round produced only ONE billing example -- far too few
                         to measure the thing that matters most.
  adversarial          - sarcasm, shouting, multi-question, very short, very
                         long. Not representative; included to find where the
                         system breaks rather than to estimate average quality.

Only conversation openers are sampled: a mid-thread reply ("yes, during ads")
carries its meaning in an earlier turn the agent never sees.

Existing hand-labelled rows are always preserved -- this script appends new
unlabelled rows, it never discards human work.
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_prep import ROOT

TARGET_TOTAL = 200
SEED = 7

QUOTA = {"natural": 90, "rare_boost": 40, "escalation_sensitive": 35, "adversarial": 35}

# Escalation-sensitive language: money, account access, security, legal.
SENSITIVE = re.compile(
    r"\b(charge[ds]?|charging|bill(ed|ing)?|refund|credit|money|paid|payment|"
    r"card|invoice|subscription|cancel(l?ed|l?ing)?|password|log ?in|login|"
    r"logged out|account|hack(ed)?|fraud|unauthorh?i[sz]ed|stolen|dispute|"
    r"chargeback|lawyer|legal)\b", re.I)

SARCASM = re.compile(r"\b(thanks a lot|great job|nice (job|work)|love how|so helpful|"
                     r"way to go|congrats|brilliant|genius)\b", re.I)
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


def adversarial_flags(text: str) -> dict:
    letters = [c for c in text if c.isalpha()]
    caps_ratio = sum(c.isupper() for c in letters) / max(len(letters), 1)
    return {
        "very_short": len(text) < 40,
        "very_long": len(text) > 200,
        "shouting": caps_ratio > 0.6 and len(letters) > 15,
        "multi_question": text.count("?") >= 2,
        "sarcasm": bool(SARCASM.search(text)),
        "emoji_heavy": len(EMOJI.findall(text)) >= 3,
    }


def take(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return pool.sample(min(n, len(pool)), random_state=seed)


def main(confirm: bool = False) -> None:
    # This script writes to the file holding hand-made labels. Running it by
    # accident (a stray `make golden`, a mistyped target) appends unlabelled
    # rows and quietly puts the evaluation set into a half-labelled state --
    # which happened once, and was only recoverable because the labels were
    # committed. Require an explicit flag rather than trusting call sites.
    labelled_path = ROOT / "data" / "golden" / "golden_labelled.csv"
    if labelled_path.exists() and not confirm:
        existing = pd.read_csv(labelled_path)
        n_lab = int(existing["intent"].astype(str).str.strip().ne("").sum())
        raise SystemExit(
            f"REFUSING TO RUN: {labelled_path.name} already holds {n_lab} "
            f"hand-made labels.\nThis script appends UNLABELLED rows to that "
            f"file. Re-run with --confirm if that is genuinely what you want,\n"
            f"and commit the current labels first."
        )

    pool = pd.read_parquet(ROOT / "data" / "processed" / "hulu_clustered.parquet")
    pool = pool.drop_duplicates("msg_clean").reset_index(drop=True)

    labelled_path = ROOT / "data" / "golden" / "golden_labelled.csv"
    existing = pd.read_csv(labelled_path) if labelled_path.exists() else pd.DataFrame()
    already = set(existing["customer_msg"]) if len(existing) else set()
    print(f"preserving {len(already)} already-labelled examples")

    pool = pool[~pool["customer_msg"].isin(already)]

    flags = pool["msg_clean"].apply(adversarial_flags).apply(pd.Series)
    pool = pd.concat([pool, flags], axis=1)
    pool["is_adversarial"] = flags.any(axis=1)
    pool["adversarial_kinds"] = flags.apply(
        lambda r: ",".join(c for c in flags.columns if r[c]), axis=1)
    pool["is_sensitive"] = pool["msg_clean"].str.contains(SENSITIVE, na=False, regex=True)

    # How many more of each stratum do we still need?
    have = existing["stratum"].value_counts().to_dict() if len(existing) else {}
    picked = []

    natural = take(pool, max(0, QUOTA["natural"] - have.get("natural", 0)), SEED)
    natural["stratum"] = "natural"
    picked.append(natural)
    pool = pool.drop(natural.index)

    need_rare = max(0, QUOTA["rare_boost"] - have.get("rare_boost", 0))
    per_cluster = max(1, need_rare // max(pool["cluster"].nunique(), 1))
    rare_parts = [take(g, per_cluster, SEED) for _, g in pool.groupby("cluster")]
    rare = pd.concat(rare_parts).head(need_rare)
    rare["stratum"] = "rare_boost"
    picked.append(rare)
    pool = pool.drop(rare.index)

    sens = take(pool[pool["is_sensitive"]],
                max(0, QUOTA["escalation_sensitive"] - have.get("escalation_sensitive", 0)), SEED)
    sens["stratum"] = "escalation_sensitive"
    picked.append(sens)
    pool = pool.drop(sens.index)

    adv = take(pool[pool["is_adversarial"]],
               max(0, QUOTA["adversarial"] - have.get("adversarial", 0)), SEED)
    adv["stratum"] = "adversarial"
    picked.append(adv)

    new = pd.concat(picked).sample(frac=1, random_state=SEED).reset_index(drop=True)

    keep = ["customer_msg", "msg_clean", "brand_reply", "customer_tweet_id",
            "thread_root", "cluster", "stratum", "is_adversarial",
            "adversarial_kinds", "is_sensitive"]
    new = new[keep]
    for col in ["intent", "should_escalate"]:
        new[col] = ""

    combined = pd.concat([existing, new], ignore_index=True) if len(existing) else new
    out = ROOT / "data" / "golden" / "golden_labelled.csv"
    combined.to_csv(out, index=False)

    print(f"added {len(new)} new unlabelled rows -> {len(combined)} total")
    print()
    print(combined["stratum"].value_counts().to_string())
    unlabelled = combined["intent"].astype(str).str.strip().eq("").sum()
    print(f"\nstill to label: {unlabelled}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true",
                    help="acknowledge that this appends unlabelled rows to the "
                         "file containing hand-made labels")
    main(**vars(ap.parse_args()))
