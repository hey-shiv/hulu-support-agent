"""Headline metrics with bootstrap confidence intervals.

Macro-F1 is the primary intent metric, not accuracy. Accuracy on an imbalanced
set is carried by whichever intents happen to be common, while the intents that
matter most here (billing_charge, account_access) are the rarest -- exactly the
ones accuracy lets you fail silently.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_prep import ROOT

N_BOOTSTRAP = 2000
SEED = 0

INTENT_SYSTEMS = [("trivial_intent", "trivial (majority class)"),
                  ("rules_intent", "simple (keyword rules)"),
                  ("tfidf_intent", "simple (TF-IDF, silver-trained)"),
                  ("agent_intent", "agent (LLM)")]
REPLY_SYSTEMS = [("trivial", "trivial (constant apology)"),
                 ("nearest", "simple (copy nearest reply)"),
                 ("agent", "agent (grounded generation)")]
ESC_SYSTEMS = [("trivial_escalate", "trivial (always escalate)"),
               ("rules_escalate", "simple (sensitive intents only)"),
               ("agent_escalate", "agent (intent+evidence+claims)")]
DIMS = ["grounding", "correctness", "relevance", "safety", "tone", "overall"]


def boot_ci(fn, n_rows: int):
    rng = np.random.default_rng(SEED)
    idx = np.arange(n_rows)
    scores = [fn(rng.choice(idx, size=n_rows, replace=True)) for _ in range(N_BOOTSTRAP)]
    return np.percentile(scores, [2.5, 97.5])


def intent_block(df: pd.DataFrame, title: str) -> None:
    y = df["true_intent"].to_numpy()
    n = len(df)
    print(f"\n{title}  (n={n})")
    print(f"  {'system':<34}{'macro-F1':>10}{'95% CI':>20}{'accuracy':>10}")
    for col, name in INTENT_SYSTEMS:
        if col not in df:
            continue
        p = df[col].to_numpy()
        mf1 = f1_score(y, p, average="macro", zero_division=0)
        lo, hi = boot_ci(lambda s: f1_score(y[s], p[s], average="macro", zero_division=0), n)
        print(f"  {name:<34}{mf1:>10.3f}   [{lo:.3f}, {hi:.3f}]{accuracy_score(y, p):>10.3f}")


def main() -> None:
    df = pd.read_csv(ROOT / "reports" / "predictions.csv")
    n = len(df)
    print(f"n = {n} human-labelled test examples")
    if "is_thread_start" in df:
        n_open = int(df["is_thread_start"].sum())
        print(f"    {n_open} conversation openers / {n - n_open} mid-thread fragments")

    print("\n" + "=" * 74)
    print("INTENT CLASSIFICATION")
    print("=" * 74)
    intent_block(df, "ALL EXAMPLES")
    if "is_thread_start" in df and df["is_thread_start"].nunique() > 1:
        intent_block(df[df["is_thread_start"]], "OPENERS ONLY  <- the defined task")
        intent_block(df[~df["is_thread_start"]],
                     "MID-THREAD ONLY  <- unclassifiable without prior turns")

    print("\nPER-INTENT (agent, all examples):")
    print(classification_report(df["true_intent"], df["agent_intent"],
                                zero_division=0, digits=2))

    labels = sorted(set(df["true_intent"]) | set(df["agent_intent"]))
    print("CONFUSION MATRIX (rows=true, cols=predicted):")
    cm = confusion_matrix(df["true_intent"], df["agent_intent"], labels=labels)
    short = [l[:10] for l in labels]
    print(" " * 12 + "".join(f"{s:>11}" for s in short))
    for lab, row in zip(short, cm):
        print(f"{lab:>12}" + "".join(f"{v:>11}" for v in row))

    print("\n" + "=" * 74)
    print("ESCALATION  (a false auto-handle is the expensive error, not a false alarm)")
    print("=" * 74)
    y_esc = df["true_escalate"].astype(str).str.lower().eq("yes").to_numpy()
    n_pos = int(y_esc.sum())
    print(f"true escalations: {n_pos} / {n}\n")
    print(f"  {'system':<34}{'recall':>8}{'prec':>8}{'missed esc.':>13}{'needless esc.':>15}")
    for col, name in ESC_SYSTEMS:
        if col not in df:
            continue
        p = df[col].astype(str).str.lower().isin(["true", "yes", "1"]).to_numpy()
        tp, fn_, fp = int((p & y_esc).sum()), int((~p & y_esc).sum()), int((p & ~y_esc).sum())
        rec, prec = tp / max(tp + fn_, 1), tp / max(tp + fp, 1)
        missed = fn_ / max(n_pos, 1)
        needless = fp / max(int((~y_esc).sum()), 1)
        print(f"  {name:<34}{rec:>8.2f}{prec:>8.2f}{missed:>13.2f}{needless:>15.2f}")
    if n_pos < 10:
        print(f"\n  WARNING: {n_pos} positive examples. These rates are not stable;")
        print("  one flipped prediction moves recall by "
              f"{100/max(n_pos,1):.0f} percentage points.")

    if "agent_escalate_reason_code" in df:
        print("\n  which rule fired:")
        for k, v in df["agent_escalate_reason_code"].value_counts().items():
            print(f"    {k:<20}{v:>4}")

    print("\n" + "=" * 74)
    print("REPLY QUALITY  (LLM judge 1-5; see judge_validity.py / judge_agreement.py)")
    print("=" * 74)
    print(f"  {'system':<34}" + "".join(f"{d[:9]:>11}" for d in DIMS))
    for key, name in REPLY_SYSTEMS:
        cells = []
        for d in DIMS:
            col = f"judge_{key}_{d}"
            cells.append(f"{df[col].mean():>11.2f}" if col in df and df[col].notna().any()
                         else f"{'-':>11}")
        print(f"  {name:<34}" + "".join(cells))

    # The claim that matters most and is weakest: does generating a reply beat
    # simply copying the nearest historical one? Paired bootstrap on the
    # per-example difference, because the two systems answer the same messages.
    if {"judge_agent_overall", "judge_nearest_overall"} <= set(df.columns):
        a = df["judge_agent_overall"].to_numpy(dtype=float)
        b = df["judge_nearest_overall"].to_numpy(dtype=float)
        ok = ~(np.isnan(a) | np.isnan(b))
        a, b = a[ok], b[ok]
        diff = a - b
        lo, hi = boot_ci(lambda s: diff[s].mean(), len(diff))
        print(f"\n  agent overall {a.mean():.2f}  vs  copy-nearest {b.mean():.2f}")
        print(f"  paired difference: {diff.mean():+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")
        if lo <= 0 <= hi:
            print("  -> interval contains zero: generation is NOT shown to beat")
            print("     copying the nearest historical reply on this metric.")

    # NaN means "no forbidden claim found" -- an empty-string check here would
    # count str(NaN) == 'nan' as a hit and report every row as a fabrication.
    if "agent_forbidden_claim" in df:
        col = df["agent_forbidden_claim"]
        k = int(col.notna() & col.astype(str).str.strip().ne("").sum()) if col.notna().any() else 0
        print(f"\n  replies caught fabricating an account action: {k} / {n}")
    for col, label in [("agent_reply_parse_ok", "reply JSON parsed"),
                       ("agent_classify_parse_ok", "classification JSON parsed")]:
        if col in df:
            k = int(df[col].astype(str).str.lower().eq("true").sum())
            print(f"  {label}: {k} / {n}")

    print("\n" + "=" * 74)
    print("BY STRATUM  (these populations are not interchangeable)")
    print("=" * 74)
    print(f"  {'stratum':<24}{'n':>5}{'macro-F1':>11}{'accuracy':>11}")
    for stratum, g in df.groupby("stratum"):
        print(f"  {stratum:<24}{len(g):>5}"
              f"{f1_score(g['true_intent'], g['agent_intent'], average='macro', zero_division=0):>11.3f}"
              f"{accuracy_score(g['true_intent'], g['agent_intent']):>11.3f}")
    print("\n  'natural' is the only stratum that estimates production performance.")
    print("  The others are deliberately enriched and read worse by construction.")


if __name__ == "__main__":
    main()
