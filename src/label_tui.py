"""Keyboard-driven labelling tool for the golden set.

Shows no model prediction -- pre-filling a suggested label would make this
an evaluation of "does the annotator agree with the model" instead of an
independent ground truth.
"""

import sys
import termios
import tty
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.taxonomy import INTENT_NAMES

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden"
LABELLED = GOLDEN / "golden_labelled.csv"

KEYS = "1234567890"
INTENT_KEYS = dict(zip(KEYS, INTENT_NAMES))

con = Console()


def getkey() -> str:
    if not sys.stdin.isatty():
        return (sys.stdin.readline().strip() or "s")[:1]
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        k = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return "q" if k in ("\x03", "\x04") else k


def render(row, i, total) -> None:
    con.clear()
    con.print(f"[dim]{i + 1}/{total}  ·  stratum={row.stratum}[/dim]\n")
    con.print(Panel(str(row.msg_clean), title="customer message", border_style="cyan"))
    for k, name in INTENT_KEYS.items():
        con.print(f"  [cyan]{k}[/cyan]  {name}")
    con.print("\n[dim]q = save & quit[/dim]")


def main() -> None:
    src = LABELLED if LABELLED.exists() else GOLDEN / "golden_unlabelled.csv"
    df = pd.read_csv(src)
    for c in ["intent", "should_escalate", "escalation_reason"]:
        if c not in df:
            df[c] = ""
    df[["intent", "should_escalate", "escalation_reason"]] = (
        df[["intent", "should_escalate", "escalation_reason"]]
        .fillna("").astype(object))

    todo = df.index[df["intent"].astype(str).str.strip() == ""].tolist()
    if not todo:
        con.print("[green]all rows labelled[/green]")
        return

    # Label messages from the same topic cluster consecutively. Deciding
    # "playback or device?" twenty times in a row is far faster than context-
    # switching between billing, sarcasm and content questions every message.
    # Cluster is a sampling artefact, not a label, so ordering by it cannot
    # bias which intent gets chosen.
    if "cluster" in df:
        todo.sort(key=lambda i: (df.at[i, "cluster"], i))

    for n, idx in enumerate(todo):
        row = df.loc[idx]
        while True:
            render(row, n, len(todo))
            k = getkey()
            if k == "q":
                df.to_csv(LABELLED, index=False)
                con.print(f"\n[green]saved -- {len(todo) - n} left[/green]")
                return
            if k in INTENT_KEYS:
                df.at[idx, "intent"] = INTENT_KEYS[k]
                con.print("\n[bold]escalate to a human?[/bold]  [cyan]y[/cyan]/[cyan]n[/cyan]")
                esc = ""
                while esc not in ("y", "n"):
                    esc = getkey()
                df.at[idx, "should_escalate"] = "yes" if esc == "y" else "no"
                df.to_csv(LABELLED, index=False)  # save every row
                break
    con.print(f"\n[green]done -- saved {LABELLED}[/green]")


if __name__ == "__main__":
    main()
