"""Intent taxonomy for hulu_support, v1.

Derived from TF-IDF/KMeans clustering of 21k real customer messages, plus
manual reading of samples the clustering blurred together (billing, account
access, feature requests all landed in one shapeless "generic vocabulary"
cluster -- see reports/figures/hulu_clusters.png).
"""

INTENTS = {
    "playback_error": "Content fails to play: buffering, freezing, crashing, "
                      "black screen, or an error code.",
    "device_app_issue": "The app misbehaves on a specific device (Roku, Apple TV, "
                        "Smart TV) -- won't launch, missing features on that platform.",
    "live_tv_sports_issue": "A live channel or game is missing, or a live stream "
                            "cuts out mid-event.",
    "content_availability": "Asks when a title/season/episode will be available, or "
                            "why it was removed. Nothing is broken -- it's just not there.",
    "billing_charge": "A disputed or confusing charge: charged after cancelling, "
                      "during a free trial, or wants a refund.",
    "account_access": "Can't sign in, forgot password, account locked, or a "
                      "bundled account (e.g. Spotify) won't link.",
    "feature_request": "Wants something Hulu doesn't offer, or dislikes a product "
                       "change. Nothing broken -- they want it different.",
    "general_complaint": "Angry or vague, with no specific actionable problem "
                         "stated. If any concrete fault is named, use that instead.",
    "praise_chatter": "Not a support request: compliments, jokes, tagging Hulu "
                      "in unrelated conversation.",
    "other": "A real message that fits none of the above, or is too garbled "
            "to classify.",
}

INTENT_NAMES = list(INTENTS)

# Escalation trigger 1: these intents are always risky to auto-handle, no
# matter how confident the classifier is. Wrong answer here costs real money
# or account security, not just an annoyed customer.
ALWAYS_ESCALATE_INTENTS = {"billing_charge", "account_access"}

# Escalation trigger 2: no historical precedent to ground a reply in. Measured
# as the cosine similarity of the single best-matching reference exchange.
# 0.80 is the 10th percentile of top-1 similarity measured on the REFERENCE
# corpus alone (1500 probe messages queried against the index with their own
# row excluded: p5=0.772, p10=0.802, p25=0.843, p50=0.880). Reading it off the
# golden set instead -- as an earlier version did -- would mean a threshold
# tuned on the same examples used to report the final number.
# Unlike self-reported model confidence, this is a measurable property of the
# retrieval step, not a token the model chose to emit.
EVIDENCE_ESCALATE_THRESHOLD = 0.80

# Escalation trigger 3: the model's own stated confidence. Kept because it is
# free, but deliberately given a LOW threshold and treated as a weak signal:
# it is a self-reported number, not a calibrated probability. An earlier
# version made this the primary uncertainty mechanism at threshold 0.6, and
# measurement showed the model only ever emitted 0.80/0.90/0.95 -- the trigger
# never fired once. scripts/calibration.py tests whether it carries any signal
# at all rather than assuming it does.
SELF_REPORT_ESCALATE_THRESHOLD = 0.75

ESCALATION_REASONS = {
    "sensitive_intent": "Money or account access is at stake; a wrong automated "
                        "reply costs more than a handoff.",
    "unclassifiable": "The message could not be assigned a substantive intent.",
    "weak_evidence": "No sufficiently similar historical exchange exists to "
                     "ground a reply in.",
    "low_self_report": "The classifier reported low confidence in its own label.",
    "auto_handle": "Recognised intent, with close historical precedent to draw on.",
}


def should_escalate(intent: str, evidence_similarity: float,
                    self_reported_confidence: float) -> tuple[bool, str, str]:
    """Decide auto-handle vs escalate.

    Returns (escalate, reason_code, human_readable_reason).

    Triggers are checked most-severe first so the reported reason is the
    strongest one that applies, which is what a human reviewer needs to see.
    The asymmetry is deliberate: a needless escalation costs one human's time,
    a wrong automated answer about someone's money can cost far more, so every
    trigger here fails toward escalation.
    """
    if intent in ALWAYS_ESCALATE_INTENTS:
        return True, "sensitive_intent", ESCALATION_REASONS["sensitive_intent"]
    if intent == "other":
        return True, "unclassifiable", ESCALATION_REASONS["unclassifiable"]
    if evidence_similarity < EVIDENCE_ESCALATE_THRESHOLD:
        return (True, "weak_evidence",
                f"{ESCALATION_REASONS['weak_evidence']} "
                f"(best match {evidence_similarity:.2f} < {EVIDENCE_ESCALATE_THRESHOLD})")
    if self_reported_confidence < SELF_REPORT_ESCALATE_THRESHOLD:
        return (True, "low_self_report",
                f"{ESCALATION_REASONS['low_self_report']} "
                f"({self_reported_confidence:.2f} < {SELF_REPORT_ESCALATE_THRESHOLD})")
    return False, "auto_handle", ESCALATION_REASONS["auto_handle"]
