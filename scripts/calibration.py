"""Do our uncertainty signals actually predict correctness?

The escalation policy leans on two signals. Neither is trustworthy just
because it is a number, so this tests both against ground truth instead of
assuming:

  self-reported confidence - what the LLM writes in its own JSON output. An
                             earlier version treated this as a probability and
                             gated escalation on it.
  evidence similarity      - cosine similarity of the best-matching historical
                             exchange. A measured property of retrieval.

A useful signal is higher on examples the classifier got right than on ones it
got wrong. If a signal shows no separation, gating escalation on it is
security theatre and the report should say so.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_prep import ROOT


def report_signal(df: pd.DataFrame, col: str, name: str) -> None:
    correct = df["true_intent"] == df["agent_intent"]
    right, wrong = df.loc[correct, col].dropna(), df.loc[~correct, col].dropna()

    print(f"\n{name}")
    print(f"  distinct values emitted: {sorted(df[col].dropna().unique())[:12]}")
    print(f"  when classification CORRECT (n={len(right)}): mean={right.mean():.3f}")
    print(f"  when classification WRONG   (n={len(wrong)}): mean={wrong.mean():.3f}")

    if len(right) and len(wrong):
        separation = right.mean() - wrong.mean()
        print(f"  separation (correct - wrong): {separation:+.3f}")
        # Probability a random correct example scores above a random wrong one
        # (equivalent to AUC). 0.5 = the signal carries no information.
        auc = np.mean([(r > w) + 0.5 * (r == w)
                       for r in right.to_numpy() for w in wrong.to_numpy()])
        print(f"  AUC (0.50 = useless, 1.00 = perfect): {auc:.3f}")
        verdict = ("carries real signal" if auc >= 0.65 else
                   "weak signal" if auc >= 0.55 else
                   "NO usable signal -- do not gate decisions on this")
        print(f"  verdict: {verdict}")


def correlation_with_grounding(df: pd.DataFrame) -> None:
    """Test evidence similarity against what it was actually designed to predict.

    The `weak_evidence` escalation trigger claims: when no similar historical
    exchange exists, a reply cannot be well grounded. That is a claim about
    REPLY GROUNDING, not about intent accuracy -- so scoring it against intent
    correctness (above) tests a hypothesis it never made.
    """
    sub = df.dropna(subset=["agent_top_similarity", "judge_agent_grounding"])
    if len(sub) < 10:
        print("\n  (too few rows to test)")
        return
    rho, p = spearmanr(sub["agent_top_similarity"], sub["judge_agent_grounding"])
    print("\nEVIDENCE SIMILARITY vs JUDGE GROUNDING SCORE  <- the claim it actually makes")
    print(f"  Spearman rho: {rho:+.3f}  (p={p:.3f}, n={len(sub)})")

    thresh = sub["agent_top_similarity"].quantile(0.25)
    weak = sub[sub["agent_top_similarity"] <= thresh]["judge_agent_grounding"]
    strong = sub[sub["agent_top_similarity"] > thresh]["judge_agent_grounding"]
    print(f"  grounding when evidence weak  (bottom 25%): {weak.mean():.2f}")
    print(f"  grounding when evidence strong (top 75%):   {strong.mean():.2f}")
    if abs(rho) < 0.2:
        print("  verdict: NO relationship. The weak-evidence trigger is not")
        print("           supported by this data -- it escalates on a signal that")
        print("           does not predict the thing it claims to predict.")


def main() -> None:
    df = pd.read_csv(ROOT / "reports" / "predictions.csv")
    print(f"n = {len(df)} test examples")
    print(f"intent accuracy: {(df['true_intent'] == df['agent_intent']).mean():.3f}")
    print("\n--- Do these signals predict INTENT correctness? ---")

    report_signal(df, "agent_self_report", "SELF-REPORTED CONFIDENCE (model-emitted)")
    report_signal(df, "agent_top_similarity", "EVIDENCE SIMILARITY (measured from retrieval)")

    print("\n--- Does evidence similarity predict REPLY GROUNDING? ---")
    correlation_with_grounding(df)


if __name__ == "__main__":
    main()
