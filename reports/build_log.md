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
(see `report.md` Section 3 for the full breakdown). Found five real,
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
