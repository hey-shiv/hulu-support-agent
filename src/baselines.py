"""Baselines, for all three tasks.

These exist to answer one question: does the LLM system earn its complexity?
A baseline chosen to be weak would not answer it, so the keyword rules below
were written by reading real Hulu messages, and the TF-IDF classifier is given
training data (see scripts/make_silver.py) that the LLM agent never receives.
"""

import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.taxonomy import ALWAYS_ESCALATE_INTENTS


# --- Intent: trivial --------------------------------------------------------

def intent_trivial(_message: str, majority_class: str) -> str:
    """Always predict the most common intent. The floor any system must clear."""
    return majority_class


# --- Intent: simple, rule-based (needs no labelled training data) -----------

# Ordered most-specific first: the first pattern that matches wins, so
# "charged for my no commercials plan" resolves to billing rather than ads.
KEYWORD_RULES: list[tuple[str, re.Pattern]] = [
    ("billing_charge", re.compile(
        r"\b(charge[ds]?|charging|refund|credit|bill(ed|ing)?|invoice|"
        r"double[- ]?charg|money|payment|card declined)\b", re.I)),
    ("account_access", re.compile(
        r"\b(log ?in|login|logged out|sign ?in|signed out|password|"
        r"can'?t get in|locked out|my account|reset my)\b", re.I)),
    ("live_tv_sports_issue", re.compile(
        r"\b(live tv|live stream|game|nfl|nba|sec|world series|espn|"
        r"channel|broadcast|blackout)\b", re.I)),
    ("device_app_issue", re.compile(
        r"\b(roku|apple ?tv|fire ?stick|firetv|smart ?tv|xbox|playstation|ps4|"
        r"wii|chromecast|samsung|lg|app (won'?t|will not|doesn'?t) (open|load|install))\b", re.I)),
    ("playback_error", re.compile(
        r"\b(buffer(ing)?|freez(e|ing)|crash(es|ing)?|black screen|error code|"
        r"won'?t play|not playing|playback|loading|lag(ging)?|stutter)\b", re.I)),
    ("content_availability", re.compile(
        r"\b(when (will|is|are)|season \d|episode \d|new episode|available|"
        r"why (was|did you) (remove|take)|add(ing)? .* (show|series))\b", re.I)),
    ("feature_request", re.compile(
        r"\b(please add|wish|would be (nice|great)|feature|option to|"
        r"bring back|new (ui|interface|layout)|offline)\b", re.I)),
    ("praise_chatter", re.compile(r"\b(thank|thanks|love (you|hulu)|awesome|great job|❤|💚)\b", re.I)),
]


def intent_rules(message: str) -> str:
    """Keyword classifier. What a competent engineer builds in an afternoon."""
    for intent, pattern in KEYWORD_RULES:
        if pattern.search(message):
            return intent
    return "general_complaint"


# --- Intent: simple, learned ------------------------------------------------

def fit_intent_tfidf(train_df: pd.DataFrame):
    """TF-IDF + logistic regression, trained on silver (LLM-labelled) data.

    Silver labels are NOT human ground truth and are never used to score
    anything -- they exist only so this baseline has something to learn from
    without consuming scarce human labels. See scripts/make_silver.py.
    """
    vec = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1, 2))
    X = vec.fit_transform(train_df["msg_clean"])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, train_df["intent"])
    return vec, clf


def intent_tfidf_predict(message_clean: str, vec, clf) -> str:
    return clf.predict(vec.transform([message_clean]))[0]


# --- Reply baselines --------------------------------------------------------

CONSTANT_REPLY = "Sorry for the trouble! Please reach out so we can look into this for you."


def reply_trivial(_message: str) -> str:
    """One fixed reply for everything. Deliberately included because a generic
    apology scores deceptively well on fluency-oriented metrics -- if it beats
    the system, the metric is wrong, not the baseline."""
    return CONSTANT_REPLY


def reply_nearest(message: str, vectors, pairs) -> str:
    """Copy the most similar historical reply verbatim. No generation at all.

    Only meaningful because the retrieval index excludes the evaluation set --
    when it did not, this baseline was emitting the literal ground-truth reply.
    """
    from src.retrieval import retrieve
    top = retrieve(message, vectors, pairs, k=1)
    return re.sub(r"@\w+\s*", "", str(top.iloc[0]["brand_reply"])).strip()


# --- Escalation baselines ---------------------------------------------------

def escalate_trivial() -> bool:
    """Always escalate. Perfect recall, zero automation -- the null system."""
    return True


def escalate_rules(intent: str) -> bool:
    """Escalate only on sensitive intents. No evidence or fabrication checks."""
    return intent in ALWAYS_ESCALATE_INTENTS
