"""Apply AI-assisted labels to the 140 rows added in the golden-set expansion.

IMPORTANT PROVENANCE NOTE: these 140 labels were NOT produced by blind human
labelling like the original 60. They were produced by Claude (the assistant
building this repo) reading each message and applying the taxonomy directly,
under a hard time constraint that made further human labelling impossible.

This is disclosed, not hidden, via the `labelled_by` column this script adds:
  - "human"     : the original 60, labelled by Shiva Shant via src/label_tui.py
  - "ai_claude" : these 140, labelled by Claude reading src/taxonomy.py

Critically, labelling was NOT done by qwen2.5 (the generator under test) or
llama3.1 (the judge) -- using either would make the evaluation circular, since
the system would then be graded by itself. Claude is not a component of the
evaluated pipeline, so this avoids that specific failure mode, but it is still
not independent human ground truth and the report states that plainly.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_prep import ROOT
from src.taxonomy import INTENT_NAMES

# index -> (intent, should_escalate)
LABELS = {
60: ("live_tv_sports_issue", "no"), 61: ("general_complaint", "no"),
62: ("general_complaint", "yes"), 63: ("billing_charge", "yes"),
64: ("device_app_issue", "no"), 65: ("content_availability", "no"),
66: ("playback_error", "no"), 67: ("billing_charge", "yes"),
68: ("content_availability", "no"), 69: ("playback_error", "no"),
70: ("general_complaint", "no"), 71: ("billing_charge", "yes"),
72: ("billing_charge", "yes"), 73: ("billing_charge", "yes"),
74: ("playback_error", "no"), 75: ("content_availability", "no"),
76: ("praise_chatter", "no"), 77: ("playback_error", "no"),
78: ("feature_request", "no"), 79: ("playback_error", "no"),
80: ("general_complaint", "no"), 81: ("account_access", "yes"),
82: ("general_complaint", "no"), 83: ("content_availability", "no"),
84: ("account_access", "yes"), 85: ("account_access", "yes"),
86: ("device_app_issue", "no"), 87: ("live_tv_sports_issue", "no"),
88: ("live_tv_sports_issue", "no"), 89: ("feature_request", "no"),
90: ("content_availability", "no"), 91: ("general_complaint", "no"),
92: ("feature_request", "no"), 93: ("playback_error", "no"),
94: ("content_availability", "no"), 95: ("content_availability", "no"),
96: ("live_tv_sports_issue", "no"), 97: ("general_complaint", "no"),
98: ("content_availability", "no"), 99: ("content_availability", "no"),
100: ("device_app_issue", "no"), 101: ("device_app_issue", "no"),
102: ("live_tv_sports_issue", "no"), 103: ("other", "yes"),
104: ("billing_charge", "yes"), 105: ("device_app_issue", "no"),
106: ("content_availability", "no"), 107: ("live_tv_sports_issue", "no"),
108: ("billing_charge", "yes"), 109: ("account_access", "yes"),
110: ("playback_error", "no"), 111: ("playback_error", "no"),
112: ("live_tv_sports_issue", "no"), 113: ("content_availability", "no"),
114: ("feature_request", "no"), 115: ("playback_error", "no"),
116: ("content_availability", "no"), 117: ("device_app_issue", "no"),
118: ("device_app_issue", "yes"), 119: ("live_tv_sports_issue", "no"),
120: ("feature_request", "no"), 121: ("playback_error", "no"),
122: ("device_app_issue", "no"), 123: ("content_availability", "no"),
124: ("billing_charge", "yes"), 125: ("billing_charge", "yes"),
126: ("account_access", "yes"), 127: ("billing_charge", "yes"),
128: ("live_tv_sports_issue", "no"), 129: ("live_tv_sports_issue", "no"),
130: ("feature_request", "no"), 131: ("device_app_issue", "no"),
132: ("feature_request", "no"), 133: ("account_access", "yes"),
134: ("device_app_issue", "no"), 135: ("device_app_issue", "no"),
136: ("live_tv_sports_issue", "no"), 137: ("content_availability", "no"),
138: ("content_availability", "no"), 139: ("content_availability", "no"),
140: ("content_availability", "no"), 141: ("general_complaint", "no"),
142: ("billing_charge", "yes"), 143: ("feature_request", "no"),
144: ("billing_charge", "yes"), 145: ("content_availability", "no"),
146: ("feature_request", "no"), 147: ("general_complaint", "no"),
148: ("device_app_issue", "no"), 149: ("playback_error", "no"),
150: ("general_complaint", "no"), 151: ("account_access", "no"),
152: ("feature_request", "no"), 153: ("content_availability", "no"),
154: ("billing_charge", "yes"), 155: ("live_tv_sports_issue", "no"),
156: ("general_complaint", "no"), 157: ("device_app_issue", "no"),
158: ("live_tv_sports_issue", "no"), 159: ("feature_request", "no"),
160: ("playback_error", "no"), 161: ("live_tv_sports_issue", "no"),
162: ("playback_error", "no"), 163: ("billing_charge", "yes"),
164: ("device_app_issue", "no"), 165: ("content_availability", "no"),
166: ("content_availability", "no"), 167: ("general_complaint", "no"),
168: ("account_access", "yes"), 169: ("live_tv_sports_issue", "no"),
170: ("account_access", "yes"), 171: ("praise_chatter", "no"),
172: ("playback_error", "no"), 173: ("content_availability", "no"),
174: ("content_availability", "no"), 175: ("live_tv_sports_issue", "no"),
176: ("content_availability", "no"), 177: ("playback_error", "no"),
178: ("praise_chatter", "no"), 179: ("playback_error", "no"),
180: ("billing_charge", "yes"), 181: ("other", "yes"),
182: ("playback_error", "no"), 183: ("account_access", "yes"),
184: ("content_availability", "no"), 185: ("general_complaint", "no"),
186: ("feature_request", "no"), 187: ("account_access", "yes"),
188: ("live_tv_sports_issue", "no"), 189: ("live_tv_sports_issue", "no"),
190: ("playback_error", "no"), 191: ("praise_chatter", "no"),
192: ("device_app_issue", "no"), 193: ("account_access", "yes"),
194: ("billing_charge", "yes"), 195: ("device_app_issue", "no"),
196: ("account_access", "yes"), 197: ("billing_charge", "yes"),
198: ("content_availability", "no"), 199: ("device_app_issue", "no"),
}


def main() -> None:
    path = ROOT / "data" / "golden" / "golden_labelled.csv"
    df = pd.read_csv(path)

    # Provenance, added now rather than assumed: rows already labelled before
    # this script ran are "human"; everything this script touches is "ai_claude".
    if "labelled_by" not in df.columns:
        already_labelled = df["intent"].notna() & (df["intent"].astype(str).str.strip() != "")
        df["labelled_by"] = ""
        df.loc[already_labelled, "labelled_by"] = "human"

    bad = [v[0] for v in LABELS.values() if v[0] not in INTENT_NAMES]
    if bad:
        raise SystemExit(f"invalid intent(s) in LABELS: {set(bad)}")

    applied = 0
    for idx, (intent, esc) in LABELS.items():
        if idx not in df.index:
            print(f"WARNING: index {idx} not in dataframe, skipping")
            continue
        current = str(df.at[idx, "intent"]) if pd.notna(df.at[idx, "intent"]) else ""
        if current.strip():
            print(f"WARNING: row {idx} already labelled ({current!r}), not overwriting")
            continue
        df.at[idx, "intent"] = intent
        df.at[idx, "should_escalate"] = esc
        df.at[idx, "labelled_by"] = "ai_claude"
        applied += 1

    df.to_csv(path, index=False)

    remaining = df["intent"].isna() | (df["intent"].astype(str).str.strip() == "")
    print(f"applied {applied} labels")
    print(f"total labelled: {(~remaining).sum()} / {len(df)}")
    print(f"still unlabelled: {remaining.sum()}")
    print()
    print("by provenance:")
    print(df["labelled_by"].value_counts().to_string())


if __name__ == "__main__":
    main()
