"""Score every brand on how much it deflects, to avoid picking one where the
public replies carry no real resolution content."""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_prep import ROOT

# Catches "DM us", "message us", "contact us here: <link>", "report this here",
# etc. -- anything that redirects the customer off Twitter instead of solving
# the problem in the reply itself.
DEFLECT = re.compile(
    r"\b(dm|d\.m\.|direct message|private message|message us|"
    r"contact us|reach out|report this|visit us|call us|email us)\b",
    re.I,
)


def main(min_pairs: int = 2000) -> None:
    pairs = pd.read_parquet(ROOT / "data" / "processed" / "pairs.parquet")
    pairs["deflects"] = pairs["brand_reply"].str.contains(DEFLECT, na=False)

    g = pairs.groupby("brand")
    table = pd.DataFrame({
        "pairs": g.size(),
        "deflect_rate": g["deflects"].mean(),
    })
    table = table[table["pairs"] >= min_pairs]
    table = table.sort_values("deflect_rate")

    table["deflect_rate"] = (table["deflect_rate"] * 100).round(1)
    print(table.head(20).to_string())


if __name__ == "__main__":
    main()
