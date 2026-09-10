"""Judge-human agreement: what we have, and what is missing.

The assignment asks for evidence of how well the LLM judge agrees with a
human. This script reports honestly on that, in whichever of two states the
repo is in:

  - if reports/human_ratings.csv exists, it computes real per-item agreement
  - if it does not, it says so plainly and points at the weaker evidence that
    was collected instead

It does not synthesise ratings, and it does not present the substitute
evidence as if it were the real thing.
"""

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_prep import ROOT

RATINGS = ROOT / "reports" / "human_ratings.csv"
VALIDITY = ROOT / "reports" / "judge_validity.csv"


def report_gap() -> None:
    print("=" * 66)
    print("PER-ITEM JUDGE-HUMAN AGREEMENT: NOT COLLECTED")
    print("=" * 66)
    print("""
No human rated individual replies, so the strongest form of this evidence --
"on real, ambiguous replies, does the judge score them the way a person
does?" -- does not exist in this repo. That is a genuine gap against the
assignment, not an oversight, and every judge-derived number should be read
with it in mind.

What WAS collected instead (scripts/judge_validity.py):

  - the judge ranks genuine human-written Hulu replies above generic
    non-answers, and above the same reply with its actionable step removed
  - it penalises an injected fabricated refund on the safety dimension
    specifically
  - it recovers a quality ordering that was fixed by construction

Why that is weaker: those variants differ obviously. Two real replies from
two systems differ subtly, and the judge sits at the 5.00 ceiling on good
replies -- so it may not separate a better reply from a slightly worse one,
which is exactly what comparing the agent to a strong baseline requires.

Practical consequence for reading the results:
  - treat judge scores as a RANKING signal across systems, not as an
    absolute quality level
  - do not treat small judge gaps as meaningful; see the paired
    agent-vs-copy-nearest interval in the results, which contains zero

To collect the real thing: `make rate` (21 blind ratings, ~4 minutes),
then re-run this script.
""".strip())

    if VALIDITY.exists():
        df = pd.read_csv(VALIDITY)
        if "variant" in df:
            print("\nsubstitute evidence on file (reports/judge_validity.csv):")
            order = ["real", "stripped", "generic", "fabricated"]
            present = [v for v in order if v in set(df["variant"])]
            print(df.groupby("variant")["overall"].mean().reindex(present).round(2).to_string())


def report_agreement(q: pd.DataFrame) -> None:
    q["human_score"] = pd.to_numeric(q["human_score"], errors="coerce")
    q["judge_score"] = pd.to_numeric(q["judge_score"], errors="coerce")
    q = q.dropna(subset=["human_score", "judge_score"])
    n = len(q)
    print(f"n = {n} blind-rated replies\n")
    if n < 10:
        print("WARNING: fewer than 10 ratings; treat as indicative only.\n")

    rho, p = spearmanr(q["judge_score"], q["human_score"])
    diff = q["judge_score"] - q["human_score"]
    print(f"  Spearman rho (ranking):      {rho:>6.2f}   (p={p:.3f})")
    print(f"  exact agreement:             {(diff == 0).mean()*100:>5.0f}%")
    print(f"  within 1 point:              {(diff.abs() <= 1).mean()*100:>5.0f}%")
    print(f"  mean signed error:           {diff.mean():>+6.2f}")

    if diff.mean() > 0.4:
        print("\n  -> judge is systematically GENEROUS; absolute scores overstate quality")
    elif diff.mean() < -0.4:
        print("\n  -> judge is systematically HARSH relative to the human")

    print("\n  per-system (is the judge biased toward one?):")
    for src, g in q.groupby("source"):
        gap = g["judge_score"].mean() - g["human_score"].mean()
        print(f"    {src:<12} n={len(g):<3} judge={g['judge_score'].mean():.2f} "
              f"human={g['human_score'].mean():.2f}  gap={gap:+.2f}")

    print("\n  limitations: one rater, one pass, no inter-human ceiling measured,")
    print("  and the rater also built the system (blinding reduces, not removes).")


def main() -> None:
    if RATINGS.exists():
        report_agreement(pd.read_csv(RATINGS))
    else:
        report_gap()


if __name__ == "__main__":
    main()
