"""Does retrieval actually earn its place?

The central claim of this system is that replies are grounded in how Hulu
historically resolved similar problems. That claim is untested until we see
what happens without the grounding.

Three configurations, same 60 messages, same judge:

  k=0  no retrieved evidence at all -- the model answers from its own priors
  k=1  a single nearest historical exchange
  k=3  the shipped configuration

If k=0 scores close to k=3, the retrieval machinery is decoration and the
honest conclusion is that a bare prompt would do. Paired comparison, because
every configuration answers the same messages.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.agent import GEN_MODEL, classify, draft_reply, strip_handles
from src.data_prep import ROOT
from src.judge import judge_reply
from src.llm import generate_json
from src.retrieval import load_index

SEED = 0
N_BOOTSTRAP = 2000


def draft_without_evidence(message: str, intent: str) -> str:
    """Same task, same output format, no historical grounding."""
    prompt = f"""You are a Hulu customer support agent drafting a public reply.

Customer message (intent: {intent}): "{strip_handles(message)}"

Write a reply of 1-3 sentences in Hulu's support voice.

Hard rules:
- Never claim you have already done something to their account (refunded,
  credited, cancelled, reset). You have no access to any account system.
- Never promise a specific timeline or compensation.
- No @mentions or customer names.

Respond with JSON only: {{"reply": "<your reply>"}}"""
    return strip_handles(generate_json(prompt, model=GEN_MODEL).get("reply", ""))


def paired_ci(diff: np.ndarray):
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(diff))
    means = [diff[rng.choice(idx, size=len(diff), replace=True)].mean()
             for _ in range(N_BOOTSTRAP)]
    return np.percentile(means, [2.5, 97.5])


def main() -> None:
    gold = pd.read_csv(ROOT / "data" / "golden" / "golden_labelled.csv")
    gold = gold[gold["intent"].notna()].reset_index(drop=True)
    vectors, pairs = load_index()
    print(f"ablating retrieval over {len(gold)} messages "
          f"({len(gold) * 3 * 2} model calls)\n")

    out_path = ROOT / "reports" / "ablation_retrieval.csv"
    # Resume support: an earlier version only wrote at the end, so a stopped
    # run lost everything. Model calls are cached, but the loop is not free.
    done = set()
    rows = []
    if out_path.exists():
        prev = pd.read_csv(out_path)
        rows = prev.to_dict("records")
        done = set(prev["customer_msg"])
        print(f"resuming: {len(done)} already done\n")

    for i, r in gold.iterrows():
        msg = r["customer_msg"]
        if msg in done:
            continue
        print(f"[{i+1}/{len(gold)}] {strip_handles(msg)[:52]}...", flush=True)
        intent = classify(msg)["intent"]

        row = {"customer_msg": msg}
        for k in (1, 3):
            d = draft_reply(msg, intent, vectors, pairs, k=k)
            ev = [f'Customer: "{e["customer_msg"]}" -> Hulu replied: "{e["brand_reply"]}"'
                  for e in d["evidence"]]
            row[f"reply_k{k}"] = d["reply"]
            row[f"judge_k{k}"] = judge_reply(msg, d["reply"], ev).get("overall")
            if k == 3:
                row["evidence_k3"] = ev

        reply0 = draft_without_evidence(msg, intent)
        row["reply_k0"] = reply0
        # Judged against the SAME evidence as k=3, so the judge's grounding
        # criterion is applied identically -- otherwise an ungrounded reply
        # would be scored with nothing to be ungrounded against.
        row["judge_k0"] = judge_reply(msg, reply0, row["evidence_k3"]).get("overall")
        row.pop("evidence_k3")
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_path, index=False)  # save every row

    df = pd.DataFrame(rows)

    print("\n" + "=" * 64)
    print("RETRIEVAL ABLATION (judge overall, 1-5)")
    print("=" * 64)
    for k in (0, 1, 3):
        v = df[f"judge_k{k}"].dropna()
        label = "no evidence" if k == 0 else f"{k} exchange{'s' if k > 1 else ''}"
        print(f"  k={k}  ({label:<13}) mean={v.mean():.3f}  n={len(v)}")

    print("\n  paired differences vs the shipped k=3 configuration:")
    for k in (0, 1):
        sub = df.dropna(subset=[f"judge_k{k}", "judge_k3"])
        diff = (sub["judge_k3"] - sub[f"judge_k{k}"]).to_numpy(dtype=float)
        lo, hi = paired_ci(diff)
        sig = "SIGNIFICANT" if not (lo <= 0 <= hi) else "not significant"
        print(f"    k=3 minus k={k}: {diff.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  {sig}")

    print("\n  interpretation:")
    d0 = (df["judge_k3"] - df["judge_k0"]).dropna()
    lo0, hi0 = paired_ci(d0.to_numpy(dtype=float))
    if lo0 <= 0 <= hi0:
        print("    Retrieval is NOT shown to improve reply quality on this metric.")
        print("    The grounding machinery would need a different justification")
        print("    (auditability, controllability) than measured reply quality.")
    else:
        print("    Retrieval measurably improves reply quality over an ungrounded")
        print("    prompt, which is the system's central claim.")


if __name__ == "__main__":
    main()
