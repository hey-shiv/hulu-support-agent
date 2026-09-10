"""Run the agent and all baselines over the golden set, save every prediction.

All 60 human-labelled examples are used as the test set. Nothing is held back
for training or tuning, because nothing here was tuned on them:

  - the LLM agent is prompted, never trained
  - the keyword baseline was written by reading the corpus, not the golden set
  - the TF-IDF baseline trains on silver (LLM-labelled) non-golden messages
  - the evidence threshold was derived from the reference corpus, not the
    golden set

Splitting 30% off for a dev set would have cost 18 of 60 scarce human labels
to guard against tuning that does not happen. That trade is stated here so a
reviewer can disagree with it explicitly rather than discover it.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.agent import run_agent, strip_handles
from src.baselines import (escalate_rules, escalate_trivial, fit_intent_tfidf,
                           intent_rules, intent_tfidf_predict, intent_trivial,
                           reply_nearest, reply_trivial)
from src.data_prep import ROOT
from src.judge import judge_reply
from src.retrieval import load_index

JUDGE_DIMS = ["grounding", "correctness", "relevance", "safety", "tone", "overall"]


def main(limit: int | None = None) -> None:
    gold = pd.read_csv(ROOT / "data" / "golden" / "golden_labelled.csv")
    gold = gold[gold["intent"].notna() & (gold["intent"].astype(str).str.strip() != "")]
    gold = gold.reset_index(drop=True)
    print(f"test set: {len(gold)} human-labelled examples "
          f"({int(gold['is_thread_start'].sum())} conversation openers)")

    # Majority class comes from silver data, not from the labels we score against.
    silver_path = ROOT / "data" / "processed" / "silver_labels.parquet"
    if silver_path.exists():
        silver = pd.read_parquet(silver_path)
        majority_class = silver["intent"].value_counts().idxmax()
        vec, clf = fit_intent_tfidf(silver)
        print(f"baselines trained on {len(silver)} silver-labelled messages")
    else:
        raise SystemExit("run scripts/make_silver.py first")

    vectors, pairs = load_index()
    if limit:
        gold = gold.head(limit)
        print(f"SMOKE TEST: {limit} rows only")

    rows = []
    for i, r in gold.iterrows():
        msg = r["customer_msg"]
        print(f"[{i+1}/{len(gold)}] {strip_handles(msg)[:56]}...", flush=True)

        agent = run_agent(msg, vectors, pairs)
        replies = {
            "agent": agent["reply"],
            "trivial": reply_trivial(msg),
            "nearest": reply_nearest(msg, vectors, pairs),
        }

        # Same evidence shown to the judge for every candidate, so it cannot
        # favour one system by seeing more context for it.
        evidence = [f'Customer: "{e["customer_msg"]}" -> Hulu replied: "{e["brand_reply"]}"'
                    for e in agent["evidence"]]

        row = {
            "customer_msg": msg,
            "true_intent": r["intent"],
            "true_escalate": r["should_escalate"],
            "stratum": r["stratum"],
            "is_thread_start": r.get("is_thread_start", True),
            # Carried through so metrics can separate headline numbers (human
            # labels only) from the larger AI-labelled extension.
            "labelled_by": r.get("labelled_by", "human"),
            "agent_intent": agent["intent"],
            "agent_self_report": agent["self_reported_confidence"],
            "agent_top_similarity": agent["top_similarity"],
            "agent_escalate": agent["escalate"],
            "agent_escalate_reason_code": agent["escalate_reason_code"],
            "agent_escalate_reason": agent["escalate_reason"],
            "agent_forbidden_claim": agent["forbidden_claim"],
            "agent_classify_parse_ok": agent["classify_parse_ok"],
            "agent_reply_parse_ok": agent["reply_parse_ok"],
            "agent_evidence": " || ".join(evidence),
            "trivial_intent": intent_trivial(msg, majority_class),
            "rules_intent": intent_rules(msg),
            "tfidf_intent": intent_tfidf_predict(r["msg_clean"], vec, clf),
            "trivial_escalate": escalate_trivial(),
            "rules_escalate": escalate_rules(intent_rules(msg)),
        }
        for name, text in replies.items():
            row[f"{name}_reply"] = text
            scores = judge_reply(msg, text, evidence)
            for d in JUDGE_DIMS:
                row[f"judge_{name}_{d}"] = scores.get(d)
            row[f"judge_{name}_parse_ok"] = bool(scores)
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = ROOT / "reports" / "predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"\nsaved {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    main(**vars(ap.parse_args()))
