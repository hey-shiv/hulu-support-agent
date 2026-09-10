"""Load the raw Twitter support corpus."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "twcs.csv"

# Twitter's legacy timestamp format, e.g. "Tue Oct 31 22:11:45 +0000 2017".
# Giving pandas this format explicitly avoids it guessing per-row, which is slow.
TWITTER_TS = "%a %b %d %H:%M:%S %z %Y"


def load_raw(path: Path = RAW, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        nrows=nrows,
        dtype={
            "tweet_id": "int64",
            "author_id": "string",
            "text": "string",
            "in_response_to_tweet_id": "float64",  # float because some rows are blank
        },
    )
    df["created_at"] = pd.to_datetime(df["created_at"], format=TWITTER_TS)
    return df


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct (customer message -> brand reply) pairs.

    Walk backward from brand replies, since each reply already carries a pointer
    to its parent (in_response_to_tweet_id). Walking forward from customer
    messages would mean scanning the whole table per message to find who replied.
    """
    by_id = df.set_index("tweet_id")  # the "phone book" for O(1) lookups

    replies = df[(df["inbound"] == False) & df["in_response_to_tweet_id"].notna()].copy()
    replies["parent_id"] = replies["in_response_to_tweet_id"].astype("int64")

    # Some pointers reference a tweet that isn't in the table at all (deleted
    # tweet, or -- in a partial sample like our 1000-row test -- just not loaded).
    # Drop those rather than crash the lookup.
    replies = replies[replies["parent_id"].isin(by_id.index)]

    parents = by_id.loc[replies["parent_id"]]

    # A brand's reply can point at another brand tweet, not just a customer one.
    # We only want real customer -> brand exchanges.
    is_customer_parent = parents["inbound"].to_numpy() == True

    pairs = pd.DataFrame({
        "brand": replies["author_id"].to_numpy(),
        "customer_msg": parents["text"].to_numpy(),
        "brand_reply": replies["text"].to_numpy(),
        "customer_tweet_id": replies["parent_id"].to_numpy(),
        "reply_tweet_id": replies["tweet_id"].to_numpy(),
        # True when the customer message opens a conversation rather than
        # continuing one. A mid-thread reply ("yes, during ads") is
        # unclassifiable in isolation -- its context is in an earlier turn --
        # so evaluation is restricted to openers.
        "is_thread_start": parents["in_response_to_tweet_id"].isna().to_numpy(),
        # Root of the conversation, used to keep every message from the same
        # thread on the same side of the reference/evaluation split.
        "thread_root": parents["in_response_to_tweet_id"].fillna(
            pd.Series(parents.index, index=parents.index)).to_numpy(),
    })
    return pairs[is_customer_parent].reset_index(drop=True)
