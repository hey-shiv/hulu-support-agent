"""How far does the LLM judge actually track a human?

Reply-quality results in this repo are produced by an LLM judge. That number
means nothing on its own, so this anchors it against blind human ratings
(src/rate_tui.py -- the rater sees neither the producing system nor the
judge's score).

Reported together, because they answer different questions:
  Spearman        - does the judge RANK replies the way a human does?
  exact / within-1 - does it land on the same NUMBER?
  per-system means - does it systematically favour one system's output?

A judge can rank well while being miscalibrated (consistently one point
generous), which is fine for comparing systems and not fine for reading an
absolute score as "quality".
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_prep import ROOT

RATINGS = ROOT / "reports" / "human_ratings.csv"


def main() -> None:
    if not RATINGS.exists():
        raise SystemExit("No human ratings yet. Run `make rate` first "
                         "(~4 minutes, 21 blind ratings).")

    q = pd.read_csv(RATINGS)
    q["human_score"] = pd.to_numeric(q["human_score"], errors="coerce")
    q["judge_score"] = pd.to_numeric(q["judge_score"], errors="coerce")
    q = q.dropna(subset=["human_score", "judge_score"])

    n = len(q)
    print(f"n = {n} blind-rated replies\n")
    if n < 10:
        print("WARNING: fewer than 10 ratings. Treat everything below as indicative.\n")

    rho, p = spearmanr(q["judge_score"], q["human_score"])
    diff = q["judge_score"] - q["human_score"]

    print("=" * 60)
    print("AGREEMENT")
    print("=" * 60)
    print(f"  Spearman rho (ranking):      {rho:>6.2f}   (p={p:.3f})")
    print(f"  exact agreement:             {(diff == 0).mean()*100:>5.0f}%")
    print(f"  within 1 point:              {(diff.abs() <= 1).mean()*100:>5.0f}%")
    print(f"  mean signed error (judge-human): {diff.mean():>+5.2f}")
    print(f"  mean absolute error:         {diff.abs().mean():>6.2f}")

    if diff.mean() > 0.4:
        print("\n  -> The judge is systematically GENEROUS. Absolute judge scores")
        print("     overstate quality; use them for ranking systems, not as a")
        print("     quality level.")
    elif diff.mean() < -0.4:
        print("\n  -> The judge is systematically HARSH relative to the human.")

    verdict = ("strong -- judge rankings can be trusted" if rho >= 0.7 else
               "moderate -- directional only, do not read small gaps" if rho >= 0.4 else
               "WEAK -- judge-derived conclusions are not supported")
    print(f"\n  verdict: {verdict}")

    print("\n" + "=" * 60)
    print("PER-SYSTEM (is the judge biased toward one system?)")
    print("=" * 60)
    print(f"  {'system':<12}{'n':>4}{'judge':>8}{'human':>8}{'gap':>8}")
    for src, g in q.groupby("source"):
        gap = g["judge_score"].mean() - g["human_score"].mean()
        print(f"  {src:<12}{len(g):>4}{g['judge_score'].mean():>8.2f}"
              f"{g['human_score'].mean():>8.2f}{gap:>+8.2f}")

    gaps = q.groupby("source").apply(
        lambda g: g["judge_score"].mean() - g["human_score"].mean(), include_groups=False)
    if len(gaps) > 1 and (gaps.max() - gaps.min()) > 0.75:
        worst = gaps.idxmax()
        print(f"\n  -> The judge over-rates '{worst}' relative to the human more than")
        print("     it over-rates the others. Any agent-vs-baseline reply gap should")
        print("     be discounted by roughly this amount.")

    print("\n" + "=" * 60)
    print("LIMITATIONS")
    print("=" * 60)
    print(f"  - One rater, one pass, n={n}. No inter-human agreement ceiling was")
    print("    measured, so we cannot separate judge error from rater noise.")
    print("  - The rater also built the system, which is a conflict of interest")
    print("    blinding reduces but does not remove.")
    print("  - Ratings cover overall quality only, not the per-dimension scores;")
    print("    dimension-level judge claims rest on judge_validity.py instead.")


if __name__ == "__main__":
    main()
