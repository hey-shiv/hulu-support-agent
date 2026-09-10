"""Tests for the parts that can break silently.

Deliberately no test that asserts a model produced a particular answer -- that
would be testing the weather. These cover parsing, boundary conditions, the
escalation decision table, leakage, and metric correctness: the places where a
bug produces a plausible-looking wrong number instead of a crash.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import FORBIDDEN_CLAIMS, strip_handles
from src.baselines import KEYWORD_RULES, intent_rules
from src.data_prep import build_pairs
from src.llm import _extract_json
from src.taxonomy import (ALWAYS_ESCALATE_INTENTS, EVIDENCE_ESCALATE_THRESHOLD,
                          INTENT_NAMES, should_escalate)

ROOT = Path(__file__).resolve().parent.parent


# --- JSON parsing from messy model output -----------------------------------

def test_extract_json_plain():
    assert _extract_json('{"intent": "playback_error"}') == {"intent": "playback_error"}


def test_extract_json_wrapped_in_prose():
    raw = 'Sure! Here is the result:\n```json\n{"intent": "billing_charge"}\n```\nHope that helps.'
    assert _extract_json(raw) == {"intent": "billing_charge"}


def test_extract_json_nested_braces():
    raw = 'Result: {"a": {"b": 1}, "c": 2} trailing text {not json}'
    assert _extract_json(raw) == {"a": {"b": 1}, "c": 2}


def test_extract_json_unparseable_returns_empty():
    assert _extract_json("no json at all") == {}
    assert _extract_json('{"broken": ') == {}


# --- Escalation decision table ----------------------------------------------

@pytest.mark.parametrize("intent", sorted(ALWAYS_ESCALATE_INTENTS))
def test_sensitive_intents_always_escalate_regardless_of_signals(intent):
    """Strong evidence and high confidence must not override a money/account case."""
    escalate, code, _ = should_escalate(intent, evidence_similarity=0.99,
                                        self_reported_confidence=1.0)
    assert escalate is True
    assert code == "sensitive_intent"


def test_other_intent_escalates_as_unclassifiable():
    escalate, code, _ = should_escalate("other", 0.99, 1.0)
    assert escalate is True
    assert code == "unclassifiable"


def test_weak_evidence_escalates_even_for_safe_intent():
    escalate, code, _ = should_escalate("playback_error",
                                        evidence_similarity=EVIDENCE_ESCALATE_THRESHOLD - 0.01,
                                        self_reported_confidence=1.0)
    assert escalate is True
    assert code == "weak_evidence"


def test_strong_evidence_safe_intent_auto_handles():
    escalate, code, _ = should_escalate("playback_error", 0.95, 0.95)
    assert escalate is False
    assert code == "auto_handle"


def test_escalation_reason_is_always_non_empty():
    for intent in INTENT_NAMES:
        _, _, reason = should_escalate(intent, 0.9, 0.9)
        assert reason.strip(), f"empty reason for {intent}"


# --- Fabrication guard ------------------------------------------------------

@pytest.mark.parametrize("bad", [
    "I've issued your refund, sorry about that!",
    "We have already refunded the charge.",
    "This will be fixed within 24 hours.",
    "I've updated your account settings.",
])
def test_forbidden_claims_are_detected(bad):
    assert FORBIDDEN_CLAIMS.search(bad) is not None


@pytest.mark.parametrize("ok", [
    "Sorry about that! Have you tried restarting the app?",
    "Which device are you watching on?",
    "That should be available in your account settings.",
])
def test_legitimate_replies_are_not_flagged(ok):
    assert FORBIDDEN_CLAIMS.search(ok) is None


# --- Rule baseline ----------------------------------------------------------

def test_every_keyword_rule_emits_a_real_intent():
    """A rule emitting a label outside the taxonomy silently corrupts every
    metric it touches -- this caught two invalid labels on first run."""
    for intent, _ in KEYWORD_RULES:
        assert intent in INTENT_NAMES, f"rule emits unknown intent: {intent}"


def test_rule_baseline_always_returns_a_valid_intent():
    for msg in ["my card was charged twice", "", "asdfgh", "🤷", "ROKU APP BROKEN"]:
        assert intent_rules(msg) in INTENT_NAMES


def test_rule_ordering_prefers_money_over_device():
    """'charged on my roku' is a billing problem, not a device problem."""
    assert intent_rules("I was charged twice for my roku subscription") == "billing_charge"


# --- Text cleaning ----------------------------------------------------------

def test_strip_handles_removes_mentions():
    assert strip_handles("@hulu_support @115940 my app is broken") == "my app is broken"


def test_strip_handles_on_empty_and_handle_only():
    assert strip_handles("") == ""
    assert strip_handles("@hulu_support") == ""


# --- Conversation reconstruction --------------------------------------------

def test_build_pairs_links_reply_to_its_parent():
    df = pd.DataFrame({
        "tweet_id": [1, 2, 3],
        "author_id": ["cust", "brand", "brand"],
        "inbound": [True, False, False],
        "text": ["my app is broken", "sorry! try restarting", "unrelated brand tweet"],
        "created_at": pd.to_datetime(["2017-10-31", "2017-10-31", "2017-10-31"]),
        "response_tweet_id": ["2", None, None],
        "in_response_to_tweet_id": [np.nan, 1.0, np.nan],
    })
    pairs = build_pairs(df)
    assert len(pairs) == 1
    assert pairs.iloc[0]["customer_msg"] == "my app is broken"
    assert pairs.iloc[0]["brand_reply"] == "sorry! try restarting"
    assert bool(pairs.iloc[0]["is_thread_start"]) is True


def test_build_pairs_drops_dangling_parent_pointer():
    """A reply pointing at a tweet not present in the table must not crash."""
    df = pd.DataFrame({
        "tweet_id": [2],
        "author_id": ["brand"],
        "inbound": [False],
        "text": ["sorry!"],
        "created_at": pd.to_datetime(["2017-10-31"]),
        "response_tweet_id": [None],
        "in_response_to_tweet_id": [999.0],  # parent does not exist
    })
    assert len(build_pairs(df)) == 0


def test_build_pairs_ignores_brand_replying_to_brand():
    df = pd.DataFrame({
        "tweet_id": [1, 2],
        "author_id": ["brand", "brand"],
        "inbound": [False, False],
        "text": ["brand tweet", "brand reply to itself"],
        "created_at": pd.to_datetime(["2017-10-31", "2017-10-31"]),
        "response_tweet_id": ["2", None],
        "in_response_to_tweet_id": [np.nan, 1.0],
    })
    assert len(build_pairs(df)) == 0


# --- Evaluation integrity ---------------------------------------------------

@pytest.mark.skipif(not (ROOT / "data" / "processed" / "hulu_index_pairs.parquet").exists(),
                    reason="index not built")
def test_no_golden_example_appears_in_retrieval_index():
    """The leakage regression test. This exact bug shipped once."""
    gold = pd.read_csv(ROOT / "data" / "golden" / "golden_labelled.csv")
    idx = pd.read_parquet(ROOT / "data" / "processed" / "hulu_index_pairs.parquet")
    overlap = set(gold["customer_msg"]) & set(idx["customer_msg"])
    assert not overlap, f"{len(overlap)} golden examples leaked into the retrieval index"


@pytest.mark.skipif(not (ROOT / "data" / "processed" / "hulu_index_pairs.parquet").exists(),
                    reason="index not built")
def test_no_golden_thread_appears_in_retrieval_index():
    """Sibling turns from the same conversation leak the same answer."""
    gold = pd.read_csv(ROOT / "data" / "golden" / "golden_labelled.csv")
    idx = pd.read_parquet(ROOT / "data" / "processed" / "hulu_index_pairs.parquet")
    if "thread_root" not in gold:
        pytest.skip("golden set predates thread tracking")
    overlap = set(gold["thread_root"].dropna()) & set(idx["thread_root"].dropna())
    assert not overlap, f"{len(overlap)} golden conversation threads leaked into the index"


@pytest.mark.skipif(not (ROOT / "data" / "golden" / "golden_labelled.csv").exists(),
                    reason="no golden set")
def test_golden_set_has_no_duplicate_messages():
    gold = pd.read_csv(ROOT / "data" / "golden" / "golden_labelled.csv")
    assert gold["customer_msg"].duplicated().sum() == 0


# --- Metric counting edge cases ---------------------------------------------

def test_count_flagged_handles_nan_and_empty():
    """An all-NaN column means zero flags, not every row flagged.

    Regression: the original expression was `int(col.notna() & ....sum())`,
    which is `Series & int`. It silently returned 0 while every value was NaN,
    and raised TypeError the moment a real flag appeared -- i.e. it would only
    have crashed once the fabrication detector actually fired.
    """
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "scripts"))
    from metrics import count_flagged

    assert count_flagged(pd.Series([np.nan, np.nan, np.nan])) == 0
    assert count_flagged(pd.Series(["", "", ""])) == 0
    assert count_flagged(pd.Series([np.nan, "I have issued your refund", np.nan])) == 1
    assert count_flagged(pd.Series(["a", "b", np.nan, ""])) == 2


def test_empty_message_escalates_instead_of_crashing():
    """Regression: an empty message reached retrieval, where the embedding
    endpoint returns no vector and indexing it raised IndexError -- one blank
    message would take down the pipeline."""
    from src.agent import run_agent
    for msg in ["", "   ", "@hulu_support", "@hulu_support @115940  "]:
        out = run_agent(msg, vectors=None, pairs=None)
        assert out["escalate"] is True
        assert out["escalate_reason_code"] == "empty_message"
        assert out["reply"] == ""


@pytest.mark.skipif(not (ROOT / "data" / "golden" / "golden_labelled.csv").exists(),
                    reason="no golden set")
def test_golden_set_is_fully_labelled():
    """Canary for accidental re-sampling.

    scripts/sample_golden.py appends unlabelled rows to this same file. When
    that ran by accident, the first symptom was the leakage tests failing with
    a confusing '140 examples leaked' -- because the newly appended rows were
    in the index. This asserts the real problem directly.
    """
    gold = pd.read_csv(ROOT / "data" / "golden" / "golden_labelled.csv")
    unlabelled = gold["intent"].isna().sum() + gold["intent"].astype(str).str.strip().eq("").sum()
    assert unlabelled == 0, (
        f"{unlabelled} unlabelled rows in the golden set -- sample_golden.py "
        f"probably ran over it. Restore with: git checkout HEAD -- data/golden/")
