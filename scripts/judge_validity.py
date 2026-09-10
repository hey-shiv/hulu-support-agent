"""Does the judge discriminate replies of known quality?

Human agreement (scripts/judge_agreement.py) is the primary evidence, but it
costs human time and is therefore small. This is a complementary check that
needs none, on held-out exchanges the judge has never scored:

  known_good  - the real reply Hulu's support team actually sent
  generic     - a fixed apology that answers nothing
  fabricated  - the real reply with an invented refund and timeline bolted on

A judge worth using should rank known_good above generic, and should punish
`fabricated` on the safety dimension specifically -- that reply is fluent,
on-topic and helpful-sounding, and is exactly the failure a single "quality"
score waves through.

This measures discrimination, not correctness: a judge could rank these
perfectly and still disagree with humans on real, ambiguous replies. Both
numbers are reported.
"""

import re
import sys
from pathlib import Path

import pandas as pd

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


def main() -> None:
    vectors, pairs = load_index()
    gold = pd.read_csv(ROOT / "data" / "golden" / "golden_labelled.csv")

    # Substantive human replies only: a "please DM us" reply is not a
    # known-good example of anything.
    pool = pairs[
        (~pairs["brand_reply"].str.contains(DEFLECT, na=False, regex=True))
        & (pairs["brand_reply"].str.len() > 90)
        & (~pairs["customer_msg"].isin(set(gold["customer_msg"])))
    ]
    cases = pool.sample(N_CASES, random_state=SEED)
    print(f"judging {len(cases)} held-out exchanges x 3 reply variants\n")

    rows = []
    for _, c in cases.iterrows():
        msg, real = c["customer_msg"], c["brand_reply"]
        evidence = [f'Customer: "{e.customer_msg}" -> Hulu replied: "{e.brand_reply}"'
                    for _, e in retrieve(msg, vectors, pairs, k=3).iterrows()]
        variants = {
            "known_good": real,
            "generic": CONSTANT_REPLY,
            "fabricated": real + FABRICATION,
        }
        for name, text in variants.items():
            s = judge_reply(msg, text, evidence)
            rows.append({"variant": name, **{d: s.get(d) for d in
                        ["grounding", "correctness", "relevance", "safety", "tone", "overall"]}})

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "reports" / "judge_validity.csv", index=False)

    print(df.groupby("variant").mean(numeric_only=True).round(2).to_string())
    print()

    means = df.groupby("variant")["overall"].mean()
    safety = df.groupby("variant")["safety"].mean()
    checks = [
        ("ranks a real human reply above a generic non-answer",
         means.get("known_good", 0) > means.get("generic", 0)),
        ("penalises an invented refund/timeline on SAFETY",
         safety.get("fabricated", 5) < safety.get("known_good", 0)),
        ("penalises an invented refund/timeline OVERALL",
         means.get("fabricated", 5) < means.get("known_good", 0)),
    ]
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] judge {label}")

    if not checks[1][1]:
        print("\n  -> The safety dimension is not doing its job. Any grounding or"
              "\n     safety claim resting on this judge should be treated as"
              "\n     unsupported until the rubric is fixed.")


if __name__ == "__main__":
    main()
