# Build log: problems hit, and how they were solved

Chronological, real bugs encountered while building this, kept separate from
`decision_log.md` (which covers *design* choices) because "what broke and why"
is different, useful information -- and good material for explaining the code
live.

---

**1. `.groupby("cluster").apply(lambda g: g.sample(...))` silently produced
`NaN` cluster values.**
While drawing the `rare_boost` stratum of the golden set. Cause: pandas 3.0
drops the grouping column from what's passed into an `.apply()`'d function by
default, and unlike earlier pandas versions, does not allow opting back in
(`include_groups=True` raises `ValueError: include_groups=True is no longer
allowed`). The sampled sub-frames genuinely had no `cluster` column, so
concatenating them back together filled the gap with `NaN`.
Fix: replaced the `groupby().apply()` one-liner with a plain `for` loop over
`remaining.groupby("cluster")`, appending each group's sample to a list and
concatenating at the end. More lines, but doesn't depend on knowing an
internal pandas API quirk to trust it -- worth the trade for code that has to
be explained live.

**2. Embedding messages one at a time was going to take ~3.2 hours for the
full 21.7k-message Hulu corpus.**
Measured directly: a single `ollama.embed()` call took ~0.55s, and almost all
of that was fixed per-call overhead, not actual embedding compute -- proven
by batching 100 texts into one call, which measured at ~6ms/text, a ~90x
speedup. Fix: `embed_many()` batches every *uncached* text into groups of 200
before calling the model, while still caching each result individually so a
future partial re-run only pays for genuinely new text. Full-corpus embedding
then took about 2 minutes.

**3. Draft replies leaked a fake `@handle`-shaped token into the output.**
First test of `draft_reply()` produced: *"@playback_error We're sorry to hear
you're having trouble!..."* -- the model was shown historical examples full
of real `@handle` mentions (`@hulu_support`, `@115940`, etc.) and pattern-
matched that shape onto the new reply, inventing an `@` + intent-name token
that resembles one. Fix: strip `@\w+` from every historical example (and the
input message) before they reach the prompt, and explicitly instruct
"do not include any @mentions or names."

**4. Held-out intent accuracy came back low (0.40 agent, 0.15 for both
baselines) -- investigated rather than accepted at face value.**
Inspected every agent/true-label disagreement in `predictions.csv` by hand
(see `report.md` section 11, "Failure analysis," for the current version of
this exercise). Found five real,
explainable patterns rather than random noise: messages sampled mid-thread
that lack the context a human labeller had access to when reading the whole
exchange; a genuinely fuzzy `general_complaint` boundary (including one
sarcastic message read literally); `device_app_issue`/`account_access`
overlap; the model weighting "buffering" vocabulary over "live/game"
vocabulary when both appear together; and an escalation metric computed on
only 2 true-positive examples, which cannot be a stable signal at that n.
None of these were fixed in the time available -- they're documented as
findings for the failure-analysis section instead, which is more honest than
quietly re-running until a better-looking number comes out.

**5. CRITICAL -- total retrieval leakage: every evaluation example was in the
retrieval index.**
Found during a skeptical review pass, not by anything failing. The retrieval
index was built from all 21,681 Hulu exchanges; the golden set was sampled
from that same pool. Verified: 60/60 golden examples present in the index.
Practical consequence, verified directly: retrieving for a test message
returns *that exact message* at `sim=1.0000` as the top "similar past
exchange" -- so (a) the "simple" reply baseline was copying the literal
ground-truth historical reply, and (b) the agent was handed the real answer
as its top piece of grounding evidence. Every reply-quality number produced
before this fix was measuring memorisation, not generalisation, and the
agent-vs-simple-baseline reply comparison was meaningless.
Fix: split the corpus into a *reference* set (what retrieval may see) and an
*evaluation* set (the golden examples), with the golden examples and any
exchange from the same conversation thread removed from the index. Index
rebuilt, evaluation re-run from scratch.

**6. CRITICAL -- the confidence-based escalation trigger was dead code.**
`should_escalate()` escalates when classifier confidence < 0.6. Inspecting
the actual values the model produced across the test set: only three distinct
values ever appeared -- 0.80 (11x), 0.90 (7x), 0.95 (2x). The threshold was
never once crossed, so that entire "second independent safety trigger" never
fired in any evaluated case. Worse, the number is self-reported by the LLM in
its own JSON output: it is not a calibrated probability, it is a token the
model emits, and treating it as a probability was methodologically wrong.
Fix: replaced self-reported confidence with a measurable evidence-quality
signal (retrieval similarity of the best-matching historical exchange), and
added a calibration check that actually tests whether the signal predicts
correctness rather than assuming it does.

**7. 27% of golden examples are mid-thread fragments, not incoming messages.**
16 of 60 golden examples are customer messages sent *within* an existing
thread (replies to a support agent's earlier question), e.g. *"It is usually
during ads, yes."* These are unclassifiable in isolation because the context
lives in a previous turn the agent never sees -- and they were being scored
as if they were ordinary incoming messages, depressing every intent metric
for a reason that has nothing to do with classifier quality. Root cause: the
simplified `data_prep.py` used in the rebuild dropped the `is_thread_start`
flag that an earlier version had, so the sampler drew from all customer
messages rather than conversation openers.
Fix: `is_thread_start` restored in `data_prep.py`; new sampling restricted to
thread openers; existing examples tagged so metrics can be reported on the
clean subset with the mid-thread cases analysed separately as a documented
context-loss failure mode.

**8. (Earlier session, same project, before a from-scratch rebuild) a
`while esc not in "yn"` escalation-prompt loop in the labelling tool never
actually re-prompted.**
`"" in "yn"` evaluates to `True` in Python (an empty string is considered
"contained in" any string), so the loop's exit condition was satisfied
immediately by an unset value, and the escalation question was silently
skipped for every row -- it would have recorded every single label as "no"
without ever asking. Caught by a scripted end-to-end test (simulated
keypresses through the tool) before any real labelling happened. Fixed by
checking membership against a tuple, `("y", "n")`, not a string. The lesson
carried forward into this session's rebuild of the same tool, which used the
tuple form from the first draft.

---

**9. `sample_golden.py` silently overwrote the hand-labelled evaluation set.**
Discovered when two leakage tests failed with "140 golden examples leaked into
the retrieval index" -- a confusing symptom, because the real problem was not
leakage at all. `sample_golden.py` appends new unlabelled rows to
`golden_labelled.csv`, the same file holding the human labels, so an
accidental run put the evaluation set into a half-labelled 200-row state and
the newly appended rows were legitimately in the index. Recoverable only
because the labels had been committed.
Fixes, all three: the script now refuses to run over an existing labelled file
without `--confirm`; the `golden` target was removed from `make all` (it had
been a prerequisite, which would have destroyed labels on any full rebuild);
and a canary test asserts the golden set contains no unlabelled rows, so the
next occurrence fails with the actual cause instead of a leakage red herring.

**10. Backticks inside a Makefile `@echo` were shell command substitution.**
`@echo "human-in-the-loop (not run by `all`):"` -- make hands each recipe line
to a shell, which treats backticks as "run this and substitute the output".
The help text was executing `all` as a command rather than printing it.
Harmless here only because no command named `all` exists.

**11. An empty customer message crashed the whole pipeline.**
`run_agent("")` reached retrieval, where the embedding endpoint returns no
vector and indexing `[0]` raised IndexError. Found by deliberately feeding
degenerate inputs (empty, whitespace, handles-only, emoji-only, single
character, very long) rather than by anything failing in normal use. Now
short-circuits to escalation with reason code `empty_message`; the other five
degenerate inputs already behaved sensibly (all escalated).

**12. The fabricated-claim counter was `Series & int` and would crash on first
use.** `int(col.notna() & col.astype(str).str.strip().ne("").sum())` -- `&`
binds looser than the method chain, so this ANDs a Series with a scalar. It
printed a correct-looking `0` only because zero fabrications had been detected
and a preceding `.any()` check short-circuited. The first time the detector
actually fired, `make metrics` would have raised TypeError. Extracted to a
tested `count_flagged()` helper.

**13. CRITICAL -- 101 of 140 second-batch golden labels were mismatched to
their message content.** After the golden set was expanded from 60 to 200
examples and hand-labelled in a second session (done quickly, under real time
pressure), evaluating gave macro-F1 0.222 against the first batch's 0.487 --
a plausible-looking drop, since the new batch deliberately added categories
(`billing_charge`, `account_access`) the first batch barely sampled. It was
tempting to accept this as the expected effect of a harder, more honest
sample and move on.
Investigated instead of accepted, starting from a concrete oddity: the
`billing_charge` count sat at 12 in a 200-row set, and reading those 12
messages directly showed most had nothing to do with billing --
`billing_charge` was assigned to "Are the new Christmas Movies going to be
on?", "Does anyone like the new interface? Beautiful and terribly
unpleasant to use.", and "Trying to watch 2006 version of Penelope and end up
getting 1966's version." None of these mention money, a charge, a
subscription, or an account.
Widened the check to all 140 second-batch rows, dumping every message next
to its label and reading each one against the taxonomy definitions in
`src/taxonomy.py`. Found the same pattern throughout: labels that were
internally consistent-looking (a real intent name, a real escalate value) but
disconnected from the actual message content, most heavily concentrated in
`playback_error` (used for messages about account logins, billing charges,
content requests, and praise) and `content_availability` (used for messages
about live-TV outages, device crashes, and billing disputes). This is
consistent with a labelling session where key presses drifted from the
message on screen -- plausible under real time pressure with the keyboard-
driven, no-mouse label tool -- rather than random noise or a code bug: the
original 60-row batch, labelled unhurried, showed no comparable pattern on
the same spot-check method.
Fix: read all 140 second-batch messages against the taxonomy a second time
and corrected every mismatch found -- 101 of 140 rows (72%), including 9 of
the original 12 `billing_charge` labels. `data/golden/golden_labelled.csv`
in git history preserves both the corrupted and corrected versions for audit.
Re-ran the full pipeline (`build_index.py`, `run_eval.py`, `metrics.py`) on
the corrected set: macro-F1 0.606, accuracy 63.0% -- higher than either the
first batch alone (0.487) or the corrupted full set (0.222), and now with
adequate per-class sample sizes to trust the `billing_charge` (n=18) and
`account_access` (n=17) numbers specifically, which is the entire reason the
second batch was drawn. See `decision_log.md` entry 15 and `report.md`
section 4 for the reporting consequences of this.
