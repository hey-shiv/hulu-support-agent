"""The agent: classify intent, draft a grounded reply, decide escalation.

Every stage returns its inputs and intermediate signals alongside its output,
so any produced reply can be audited after the fact: which historical
exchanges were retrieved, how similar they were, what intent was assigned,
and which rule drove the escalation decision.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llm import generate_json
from src.retrieval import retrieve
from src.taxonomy import INTENTS, INTENT_NAMES, should_escalate

GEN_MODEL = "qwen2.5:7b-instruct-q4_K_M"

HANDLE = re.compile(r"@\w+\s*")

# Things a support reply must never invent. The model has no access to any
# account system, so any concrete claim about one is necessarily fabricated.
FORBIDDEN_CLAIMS = re.compile(
    r"\b(i(?:'ve| have) (?:issued|processed|refunded|credited|cancelled|canceled|reset)|"
    r"(?:we|i)(?:'ve| have) (?:already )?(?:refunded|credited|fixed|resolved)|"
    r"your (?:refund|credit) (?:has been|is being) (?:issued|processed)|"
    r"within \d+ (?:hours?|days?|weeks?)|"
    r"by (?:tomorrow|tonight|monday|tuesday|wednesday|thursday|friday)|"
    r"i(?:'ve| have) (?:updated|changed|removed) your)\b", re.I)


def strip_handles(text: str) -> str:
    return HANDLE.sub("", str(text)).strip()


def classify(message: str) -> dict:
    """Assign an intent. Returns the label plus the raw signals behind it."""
    clean = strip_handles(message)
    if not clean:
        return {"intent": "other", "self_reported_confidence": 0.0,
                "parse_ok": True, "note": "empty message after cleaning"}

    taxonomy_desc = "\n".join(f"- {name}: {desc}" for name, desc in INTENTS.items())
    prompt = f"""Classify this customer support message sent to Hulu into exactly one intent.

Intents:
{taxonomy_desc}

Message: "{clean}"

Rules:
- Choose "other" if no intent above genuinely fits.
- Choose the most specific intent that applies, not the most general.

Respond with JSON only: {{"intent": "<intent name>", "confidence": <0.0-1.0>}}"""

    result = generate_json(prompt, model=GEN_MODEL)
    raw_intent = result.get("intent")
    parse_ok = bool(result) and raw_intent in INTENT_NAMES

    return {
        "intent": raw_intent if parse_ok else "other",
        # Self-reported, NOT a calibrated probability. See taxonomy.py and
        # scripts/calibration.py -- this is measured, not assumed to be useful.
        "self_reported_confidence": float(result.get("confidence", 0.0)) if parse_ok else 0.0,
        "parse_ok": parse_ok,
        "note": "" if parse_ok else f"unparseable or unknown intent: {raw_intent!r}",
    }


def draft_reply(message: str, intent: str, vectors, pairs, k: int = 3) -> dict:
    """Draft a reply grounded in retrieved historical exchanges."""
    examples = retrieve(message, vectors, pairs, k=k)
    evidence = [
        {"customer_msg": strip_handles(r.customer_msg),
         "brand_reply": strip_handles(r.brand_reply),
         "similarity": float(r.similarity)}
        for _, r in examples.iterrows()
    ]

    example_text = "\n\n".join(
        f'Customer: "{e["customer_msg"]}"\nHulu support replied: "{e["brand_reply"]}"'
        for e in evidence
    )
    prompt = f"""You are a Hulu customer support agent drafting a public reply.

Here is how Hulu handled the most similar past cases:

{example_text}

New customer message (intent: {intent}): "{strip_handles(message)}"

Write a reply of 1-3 sentences, in the same voice as the examples above.

Hard rules:
- Only suggest steps or policies that appear in the examples above.
- Never claim you have already done something to their account (refunded,
  credited, cancelled, reset). You have no access to any account system.
- Never promise a specific timeline or compensation.
- If the examples do not cover this problem, ask a clarifying question instead
  of guessing.
- No @mentions or customer names.

Respond with JSON only: {{"reply": "<your reply>"}}"""

    result = generate_json(prompt, model=GEN_MODEL)
    reply = strip_handles(result.get("reply", ""))

    violation = FORBIDDEN_CLAIMS.search(reply)
    return {
        "reply": reply,
        "evidence": evidence,
        "top_similarity": evidence[0]["similarity"] if evidence else 0.0,
        "parse_ok": bool(result) and bool(reply),
        "forbidden_claim": violation.group(0) if violation else "",
    }


def run_agent(message: str, vectors, pairs, k: int = 3) -> dict:
    """Full pipeline, returning every intermediate signal for auditing."""
    cls = classify(message)
    drafted = draft_reply(message, cls["intent"], vectors, pairs, k=k)

    escalate, reason_code, reason = should_escalate(
        intent=cls["intent"],
        evidence_similarity=drafted["top_similarity"],
        self_reported_confidence=cls["self_reported_confidence"],
    )

    # A reply that broke the no-fabrication rules is never safe to auto-send,
    # whatever the intent-level decision said.
    if drafted["forbidden_claim"]:
        escalate, reason_code = True, "forbidden_claim"
        reason = f"Draft contained an unsupported claim: {drafted['forbidden_claim']!r}"

    return {
        "message": message,
        "intent": cls["intent"],
        "self_reported_confidence": cls["self_reported_confidence"],
        "classify_parse_ok": cls["parse_ok"],
        "reply": drafted["reply"],
        "reply_parse_ok": drafted["parse_ok"],
        "top_similarity": drafted["top_similarity"],
        "evidence": drafted["evidence"],
        "forbidden_claim": drafted["forbidden_claim"],
        "escalate": escalate,
        "escalate_reason_code": reason_code,
        "escalate_reason": reason,
    }
