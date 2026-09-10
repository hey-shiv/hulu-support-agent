"""Is the LLM judge trustworthy? Two tests, neither requiring new human labour.

The assignment asks for evidence of judge-human agreement. Per-item human
ratings were not collected (see reports/report.md sec.9 -- this is a stated
gap, not a solved problem). What follows is weaker but real, and its
limitations are spelled out rather than glossed:

TEST 1 - discrimination. Three variants of the same exchange, one of which is
         a genuine human-written Hulu reply. Does the judge rank them sensibly,
         and does the safety dimension catch an injected fabrication?

TEST 2 - graded ordering. Four variants whose quality ordering is fixed by
         construction (a real reply is better than the same reply with its
         actionable step removed, which is better than a generic apology,
         which is better than a reply carrying an invented refund). Does the
         judge's score recover that ordering?

What test 2 IS: agreement with a human-specified quality ordering, on replies
whose relative quality is not in dispute.
What it is NOT: agreement with a human's per-item judgment on real, ambiguous
replies -- which is the harder thing, and the thing still missing. A judge can
pass both tests here and still disagree with a person about whether a
particular real reply is a 4 or a 5.
"""

import re
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.baselines import CONSTANT_REPLY
from src.data_prep import ROOT
from src.judge import judge_reply
from src.retrieval import load_index, retrieve

N_CASES = 25
SEED = 3
DEFLECT = re.compile(r"\b(dm|direct message|private message|call us|chat)\b", re.I)
FABRICATION = (" I've gone ahead and issued a full refund to your account, "
               "and it will be fixed within 24 hours.")

# Sentences that carry the actual help: a step, a link, or an instruction.
ACTIONABLE = re.compile(
    r"(https?://|\b(try|check|restart|reboot|unplug|go to|head to|select|click|"
    r"tap|update|reinstall|make sure|ensure|visit|see)\b)", re.I)

# Known-correct quality ordering, best first. Test 2 asks whether the judge
# recovers this ranking, not whether it agrees with any particular number.
QUALITY_RANK = {"real": 0, "stripped": 1, "generic": 2, "fabricated": 3}


def strip_actionable(reply: str) -> str:
    """Remove the sentences that contain the actual help, keep the pleasantries.

    Produces a reply that is still fluent, on-topic and polite but no longer
    tells the customer what to do -- a degradation a person would agree is
    worse, which a fluency-oriented metric might not notice.
    """
    sentences = re.split(r"(?<=[.!?])\s+", str(reply))
    kept = [s for s in sentences if not ACTIONABLE.search(s)]
    return " ".join(kept).strip() or "Sorry to hear that!"


def main() -> None:
    vectors, pairs = load_index()
    gold = pd.read_csv(ROOT / "data" / "golden" / "golden_labelled.csv")

    pool = pairs[
        (~pairs["brand_reply"].str.contains(DEFLECT, na=False, regex=True))
        & (pairs["brand_reply"].str.len() > 90)
        & (pairs["brand_reply"].str.contains(ACTIONABLE, na=False, regex=True))
        & (~pairs["customer_msg"].isin(set(gold["customer_msg"])))
    ]
    cases = pool.sample(N_CASES, random_state=SEED)
    print(f"judging {len(cases)} held-out exchanges x 4 variants "
          f"({len(cases) * 4} judge calls)\n")

    rows = []
    for _, c in cases.iterrows():
        msg, real = c["customer_msg"], c["brand_reply"]
        evidence = [f'Customer: "{e.customer_msg}" -> Hulu replied: "{e.brand_reply}"'
                    for _, e in retrieve(msg, vectors, pairs, k=3).iterrows()]
        variants = {
            "real": real,
            "stripped": strip_actionable(real),
            "generic": CONSTANT_REPLY,
            "fabricated": real + FABRICATION,
        }
        for name, text in variants.items():
            s = judge_reply(msg, text, evidence)
            rows.append({"variant": name, "rank": QUALITY_RANK[name],
                         **{d: s.get(d) for d in ["grounding", "correctness",
                            "relevance", "safety", "tone", "overall"]}})

    df = pd.DataFrame(rows).dropna(subset=["overall"])
    df.to_csv(ROOT / "reports" / "judge_validity.csv", index=False)

    print("mean score by variant (quality ordering is real > stripped > generic > fabricated):")
    order = ["real", "stripped", "generic", "fabricated"]
    print(df.groupby("variant").mean(numeric_only=True).reindex(order).round(2).to_string())

    print("\n" + "=" * 62)
    print("TEST 1 - DISCRIMINATION")
    print("=" * 62)
    m = df.groupby("variant")["overall"].mean()
    safety = df.groupby("variant")["safety"].mean()
    for label, ok in [
        ("ranks a real human reply above a generic non-answer",
         m.get("real", 0) > m.get("generic", 0)),
        ("notices when the actionable step is removed",
         m.get("real", 0) > m.get("stripped", 0)),
        ("penalises an invented refund on SAFETY",
         safety.get("fabricated", 5) < safety.get("real", 0)),
    ]:
        print(f"  [{'PASS' if ok else 'FAIL'}] judge {label}")

    print("\n" + "=" * 62)
    print("TEST 2 - AGREEMENT WITH A HUMAN-SPECIFIED QUALITY ORDERING")
    print("=" * 62)
    # Negative rho expected: rank 0 is best, so score should fall as rank rises.
    rho, p = spearmanr(df["rank"], df["overall"])
    print(f"  Spearman rho (rank vs judge score): {rho:+.3f}  (p={p:.4f}, n={len(df)})")
    print(f"  |rho| = {abs(rho):.3f}   (1.00 = ordering perfectly recovered)")

    # Pairwise: over all (better, worse) pairs, how often does the judge agree?
    correct = total = 0
    for a in order:
        for b in order:
            if QUALITY_RANK[a] < QUALITY_RANK[b]:
                sa = df[df["variant"] == a]["overall"].to_numpy()
                sb = df[df["variant"] == b]["overall"].to_numpy()
                n = min(len(sa), len(sb))
                correct += int((sa[:n] > sb[:n]).sum())
                total += n
    print(f"  pairwise ordering accuracy: {correct}/{total} = {correct/max(total,1)*100:.0f}%")

    verdict = ("strong" if abs(rho) >= 0.7 else
               "moderate" if abs(rho) >= 0.4 else "weak")
    print(f"  verdict: {verdict}")

    print("\n" + "=" * 62)
    print("WHAT THIS DOES NOT SHOW")
    print("=" * 62)
    print("  These variants differ obviously. Real replies from different")
    print("  systems differ subtly, and the table above shows the judge sitting")
    print("  at the 5.00 ceiling for good replies -- so it may not separate a")
    print("  genuinely better reply from a slightly worse one. Per-item human")
    print("  ratings on real replies remain the missing evidence.")


if __name__ == "__main__":
    main()
