# Hulu Support Agent — Report

## 1. Executive summary

An AI support agent for `hulu_support` that classifies an incoming customer
message into one of 10 intents, drafts a reply grounded in retrieved
historical Hulu resolutions, and decides auto-handle vs. escalate with a
stated reason.

Evaluated on **60 hand-labelled examples** against **three baselines**.

**Headline: macro-F1 0.673 on the defined task (conversation openers),
vs. 0.362 for TF-IDF and 0.054 for majority-class.** Intervals do not overlap.

**Three findings that argue against this system, which matter more than the
headline:**

1. **Retrieval earns its place; the generation layer on top of it does not.**
   Removing retrieval entirely costs 0.87 judge points (CI [+0.67, +1.08]) —
   grounding is doing real work. But generating a reply does not beat simply
   copying the nearest historical reply (+0.12, CI [−0.10, +0.35], contains
   zero). The value is in the retrieval, not the LLM wrapped around it.
2. **On mid-thread messages the agent is *worse* than TF-IDF** (0.190 vs
   0.240). Fluency becomes a liability when context is missing.
3. **The model's self-reported confidence is worthless** — AUC 0.539 for
   predicting its own correctness. An earlier version of this system gated
   escalation on it.
4. **The shipped retrieval depth is probably wrong.** k=1 significantly beats
   the shipped k=3. Not acted on, because that was measured on the test set
   and changing it would be tuning on the evaluation data.

The most important sections here are §12 (why not to trust the headline) and
§4 (two evaluation bugs found by attacking our own results — one of which
invalidated every reply-quality number produced before it).

---

## 2. Problem framing

**What "good" means for this brand.** Hulu's public support is mostly
troubleshooting that repeats: playback failures, device-specific app problems,
live-TV/sports outages, content-availability questions. A good reply therefore
(a) proposes a step Hulu has actually used before for this class of problem,
(b) does not invent policy, and (c) recognises when it must not answer at all.

Escalation quality matters at least as much as reply quality. The cost is
asymmetric: a needless handoff wastes one agent's minutes; a confident wrong
answer about someone's money or account access can cost a customer, a refund
dispute, or a public complaint. Every escalation rule here therefore fails
*toward* escalation.

**What was NOT built, and why:**

- **Multi-turn dialogue.** The agent answers one incoming message. Mid-thread
  follow-ups ("yes, during ads") need prior turns to mean anything; rather
  than half-support them, they are excluded from the defined task and
  measured separately (§11, failure mode 1).
- **Fine-tuning.** Retrieval satisfies the "grounded in how this brand
  historically resolved it" requirement directly, stays inspectable, and
  updates when the corpus does. Fine-tuning would bake 2017 policy into
  weights and remove the audit trail.
- **Any agent/RAG framework.** Retrieval is a cosine similarity search over a
  numpy array. Every step is readable and explainable.
- **A production safety layer** beyond the escalation policy and the
  fabricated-claim regex. Real deployment would need PII handling and abuse
  detection.
- **Judge rubric tuning.** The rubric was written once from the failure modes
  we cared about, then validated (§9). Iterating it against results would have
  meant tuning the measuring instrument to flatter the thing measured.

---

## 3. Architecture

```
incoming message
      |
      +--> classify ------------> intent (10 classes + "other")
      |                           + self-reported confidence  [weak signal]
      |
      +--> retrieve ------------> 3 nearest historical Hulu exchanges
      |                           + top-1 similarity          [evidence quality]
      |
      +--> draft ---------------> reply constrained to that evidence
      |                           + fabricated-claim check
      |
      +--> escalate? -----------> auto-handle | escalate + reason code
```

Escalation fires on the first matching rule: sensitive intent
(billing/account) → unclassifiable (`other`) → weak evidence (top-1
similarity < 0.80) → low self-report. Separately, a fabricated claim detected
in the drafted reply **overrides** whatever that chain decided — a reply that
invents an account action is never safe to auto-send regardless of intent.

---

## 4. Data and evaluation integrity

**Source.** Kaggle `thoughtvector/customer-support-on-twitter`, 2.8M tweets.
Reconstructed into 1.26M `(customer message → brand reply)` exchanges by
walking backward from each reply's parent pointer.

**Brand: `hulu_support`.** Chosen on measured deflection rate — how often a
"reply" just redirects the customer elsewhere instead of resolving anything:

| Brand | Exchanges | Deflection rate |
|---|---:|---:|
| TMobileHelp | 34,215 | 82.4% |
| AppleSupport | 106,646 | 55.7% |
| Uber_Support | 56,160 | 45.4% |
| AmazonHelp | 168,814 | 10.8% |
| **hulu_support** | **21,681** | **7.4%** |

Volume was a trap. A reply-drafting agent grounded in history is bounded by
what that history contains; grounding on "please DM us" teaches deflection.

**Two evaluation bugs found by attacking our own results:**

1. **Total retrieval leakage.** The index originally contained all 21,681
   exchanges — including every golden example. Verified: querying with a test
   message returned *that message* at similarity `1.0000`, so the
   copy-nearest-reply baseline emitted the literal ground-truth reply and the
   agent received the real answer as "evidence". Every reply-quality number
   produced before this measured memorisation.
   **Fixed:** the index excludes every golden example *and every message from
   the same conversation thread* (88 rows). Two tests fail if it regresses.

2. **The confidence escalation trigger was dead code.** It escalated below
   0.6 confidence; the model only ever emitted 0.80/0.90/0.95, so it never
   fired once — and that number is a token the model chose, not a calibrated
   probability.
   **Fixed:** escalation now gates on measured retrieval evidence quality,
   with the threshold derived from the *reference corpus* (p10 of top-1
   similarity = 0.80), not from the golden set. `scripts/calibration.py` tests
   whether either signal predicts correctness rather than assuming it.

**Other integrity measures.** Golden labels made by a human with no model
suggestion displayed. Silver (LLM) labels exist only to train the TF-IDF
baseline and are never scored against. Thresholds derived from reference data.
`temperature=0`, fixed seeds, every call cached.

---

## 5. Intent taxonomy

Derived by clustering 14,080 conversation openers (TF-IDF + KMeans), then
collapsed by hand. The clustering was informative but not usable directly: one
cluster held 51% of all messages (generic vocabulary, no coherent topic) and
another was simply the show name "Rick and Morty" — vocabulary clustering
groups by shared words, not shared meaning.

| Intent | Qualifies | Does not |
|---|---|---|
| `playback_error` | buffering, freezing, crashes, error codes | app won't launch at all → `device_app_issue` |
| `device_app_issue` | app broken on a named device (Roku, Apple TV) | stream problems on a working app |
| `live_tv_sports_issue` | live channel/game missing, stream drops mid-event | on-demand playback problems |
| `content_availability` | "when is season 3", removed titles | content that exists but won't play |
| `billing_charge` | disputed charge, refund, charged after cancelling | forward-looking plan questions |
| `account_access` | login, password, locked out, bundle won't link | app works but content missing |
| `feature_request` | wants something Hulu doesn't offer | something that used to work and broke |
| `general_complaint` | anger with no actionable specific | any message naming a concrete fault |
| `praise_chatter` | compliments, jokes, non-support | complaints phrased politely |
| `other` | genuine but fits nothing above, or garbled | anything a real class plausibly fits |

`other` is deliberate: forcing every message into a substantive class converts
"I don't know" into a confident wrong answer. `other` also triggers escalation.

---

## 6. Golden evaluation set

**60 hand-labelled examples** (44 conversation openers, 16 mid-thread).

**Sampling.** Two tagged strata: `natural` (40, plain random — the only slice
that estimates production performance) and `rare_boost` (20, spread evenly
across the 10 clusters so low-frequency intents have any measurable per-class
score at all). Every row keeps its stratum tag; results are reported per
stratum rather than blended, because blending would hide which population a
number describes.

**Labelling.** One human, one pass, using the definitions in §5, with no model
prediction ever displayed — a shown suggestion anchors annotator judgment and
would make the golden set a measurement of the model instead of a check on it.
Escalation was judged per message, not derived from intent, which is why 3
labels escalate outside the always-escalate categories.

**This is below the assignment's stated 150–250 range, and that is a real
shortfall, not an oversight.** Labelling is human work and the time was not
available. The consequences are quantified rather than hidden:

- confidence intervals on every metric are wide (§10)
- only 7 of 60 examples are true escalations, so escalation rates are unstable
- exactly 1 `billing_charge` example, so the highest-cost intent is
  effectively unmeasured

An expanded sampler (`scripts/sample_golden.py`) is built and ready, including
`escalation_sensitive` and `adversarial` strata designed specifically to fix
the second and third points. It needs roughly 40 more minutes of labelling.

---

## 7. Evaluation method

| Metric | Definition |
|---|---|
| **Macro-F1** (primary) | Unweighted mean of per-class F1. Primary because accuracy is carried by common intents while the ones that matter most are rarest. |
| Accuracy | Reported alongside, never alone. |
| Per-class P/R/F1 + confusion matrix | Where the aggregate hides failure. |
| **Missed escalation rate** | Of messages that should escalate, the fraction auto-handled. The expensive error. |
| **Needless escalation rate** | Of messages that need not escalate, the fraction escalated. The cost of caution. |
| Judge dimensions | grounding, correctness, relevance, safety, tone, overall (1–5), each separately defined. |
| Bootstrap 95% CI | 2000 resamples with replacement. |

---

## 8. Baselines

| | What it is | Why included |
|---|---|---|
| **Trivial** | Always predict the majority intent; one fixed apology for every reply; always escalate | The floor. "Always escalate" has perfect recall and zero automation — if it wins, the metric is wrong. |
| **Simple (rules)** | Hand-written keyword classifier; copy the nearest historical reply verbatim; escalate on sensitive intents | Independent of the LLM. What a competent engineer builds in an afternoon. |
| **Simple (TF-IDF)** | TF-IDF + logistic regression trained on 400 silver-labelled messages | Answers "can a microsecond-latency linear model recover the LLM's behaviour?" **Circular by construction** — it distills the LLM's own labels, so it cannot meaningfully exceed it. Stated, not hidden. |

The TF-IDF baseline is given labelled training data the LLM agent never
receives. That biases the comparison against our own system deliberately.

---

## 9. Is the judge trustworthy?

Reply-quality numbers come from an LLM judge (llama3.1) scoring replies from a
different model family (qwen2.5), so it is not grading its own style. That
helps but proves nothing, so the judge was tested directly.

**What the assignment asks for and what is missing.** The brief asks for
evidence of how well the judge agrees with a human. **Per-item human ratings
were not collected.** That is a real gap. What follows is weaker evidence,
labelled as such throughout, and `scripts/judge_agreement.py` prints an
explicit statement of the gap rather than quietly substituting this for it.

**Graded degradation test** (`scripts/judge_validity.py`, 25 held-out
exchanges x 4 variants = 100 judge calls, no human labour). Each real exchange
is scored in four variants whose quality ordering is fixed by construction:

| Variant | grounding | correctness | relevance | safety | tone | overall |
|---|---:|---:|---:|---:|---:|---:|
| real Hulu reply | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | **5.00** |
| same reply, actionable step removed | 4.36 | 4.36 | 4.52 | 4.84 | 4.68 | **4.48** |
| generic apology | 2.68 | 3.32 | 4.84 | 5.00 | 4.48 | **3.32** |
| real reply + invented refund/timeline | 1.92 | 1.56 | 4.68 | **1.00** | 3.68 | **2.16** |

**Spearman rho between the known ordering and the judge's score: -0.829
(p < 0.0001, n = 100). Pairwise ordering accuracy: 113/150 = 75%.**

All three discrimination checks pass. The judge ranks a genuine human reply
above a generic non-answer, notices the *subtle* case where a reply keeps its
tone but loses its actionable content (5.00 -> 4.48), and drives safety to the
floor (1.00) on an injected fabrication — the failure a single "quality" score
waves through.

**Three honest problems this same table exposes:**

1. **`relevance` barely works.** Spread across four wildly different reply
   qualities is 0.48 (5.00 / 4.52 / 4.84 / 4.68) — it even rates the
   fabricated reply 4.68. It contributes almost nothing and should be replaced.
2. **Ceiling effect on good replies.** Real human replies score a flat 5.00 on
   every dimension. The judge separates good from bad but may not discriminate
   *among* good replies — which is exactly what comparing our agent to a
   strong baseline requires, and is consistent with the agent-vs-copy-nearest
   comparison in §10 failing its significance test.
3. **These variants differ obviously.** Real replies from two systems differ
   subtly. Recovering a constructed ordering at rho = -0.83 does not establish
   that the judge would agree with a person about whether a particular real
   reply is a 4 or a 5.

**Practical consequence, applied throughout §10:** judge scores are treated as
a **ranking signal across systems**, never as an absolute quality level, and
small judge gaps are tested for significance rather than reported as wins.

---

## 10. Results

n = 60 human-labelled examples (44 openers, 16 mid-thread). Reproduce with
`make reproduce`.

### Intent classification — openers only (the defined task, n=44)

| System | Macro-F1 | 95% CI | Accuracy |
|---|---:|---|---:|
| Trivial (majority class) | 0.054 | [0.038, 0.083] | 0.318 |
| Simple (keyword rules) | 0.257 | [0.155, 0.378] | 0.386 |
| Simple (TF-IDF, silver-trained) | 0.362 | [0.190, 0.499] | 0.455 |
| **Agent (LLM)** | **0.673** | [0.463, 0.792] | 0.727 |

The agent beats both simple baselines with non-overlapping intervals. This is
the one result in this report that is unambiguous.

### Intent classification — mid-thread fragments (n=16)

| System | Macro-F1 | Accuracy |
|---|---:|---:|
| Simple (TF-IDF) | 0.240 | 0.375 |
| **Agent (LLM)** | **0.190** | 0.250 |

**The agent loses to TF-IDF here.** On messages whose meaning lives in a prior
turn, the LLM's fluency actively hurts — it confidently invents a reading
("it came on. Thanks for your help" → `playback_error`) where the bag-of-words
model has no strong opinion. Blended across all 60 this drags macro-F1 from
0.673 down to 0.487.

### Reply quality (LLM judge, 1–5)

| System | grounding | correctness | relevance | safety | tone | overall |
|---|---:|---:|---:|---:|---:|---:|
| Trivial (constant apology) | 2.95 | 3.52 | 4.63 | 5.00 | 4.50 | 3.55 |
| Simple (copy nearest reply) | 4.73 | 4.73 | 4.73 | 4.88 | 4.93 | 4.77 |
| Agent (grounded generation) | 4.88 | 4.90 | 5.00 | 5.00 | 5.00 | 4.88 |

**Paired difference, agent vs. copy-nearest: +0.12, 95% CI [−0.10, +0.35].**

**The interval contains zero. Generation is not shown to beat simply copying
the most similar historical reply.** Both clearly beat the constant apology,
so retrieval is earning its place — the LLM generation step on top of it is
not, on this metric, at this sample size. This is the most important negative
result in the report and it argues against the system's own complexity.

### Escalation (7 true escalations of 60)

| System | Recall | Precision | Missed escalations | Needless escalations |
|---|---:|---:|---:|---:|
| Trivial (always escalate) | 1.00 | 0.12 | 0.00 | 1.00 |
| Simple (sensitive intents) | 0.43 | 0.75 | 0.57 | 0.02 |
| Agent (intent+evidence+claims) | 0.57 | 0.25 | 0.43 | 0.23 |

The agent catches more true escalations than the rule baseline (0.57 vs 0.43)
by escalating far more often (23% vs 2% needless). Whether that trade is right
depends on a cost ratio we have not measured. **With 7 positives, one flipped
prediction moves recall by 14 points — this comparison is not statistically
meaningful and should not be used to claim the agent wins.**

Which rule fired: `auto_handle` 44, `sensitive_intent` 7, `weak_evidence` 6,
`unclassifiable` 3, `forbidden_claim` 0.

### Robustness

JSON parse success 60/60 for both classification and reply generation.
Fabricated-claim detections: 0/60 — the constrained prompt appears to be
holding, though `judge_validity.py` confirms the detector fires correctly when
a fabrication is deliberately injected.

### Does retrieval earn its place? (`scripts/ablate_retrieval.py`)

The system's central claim is that replies are grounded in Hulu's own history.
That is untested until you remove the grounding. Same 60 messages, same judge,
three configurations:

| Config | Judge overall | Paired diff vs shipped k=3 | |
|---|---:|---|---|
| k=0 — no evidence at all | 4.017 | **+0.867** [+0.667, +1.083] | significant |
| k=1 — one historical exchange | **4.983** | **−0.100** [−0.183, −0.017] | significant |
| k=3 — shipped configuration | 4.883 | — | |

**Retrieval is worth +0.87 over an ungrounded prompt**, with an interval far
from zero. This is the one central claim of the architecture that survives its
own test — and it sharpens the §10 finding: the value is in the *retrieval*,
not in the generation layer wrapped around it.

**k=1 significantly beats the shipped k=3.** Adding the 2nd and 3rd matches
made replies slightly worse, consistent with dilution — those matches are less
similar and pull the draft off-target.

**We are not switching to k=1 on the strength of this.** That difference was
measured on the same 60 examples used to report every other number here;
changing the shipped configuration because of it would be test-set tuning, the
exact practice avoided everywhere else in this project (see §4 on where the
escalation threshold came from). It is recorded as a validated hypothesis
requiring a proper dev set — see §13.

### Do the uncertainty signals work? (`scripts/calibration.py`)

| Signal | Tested against | Result |
|---|---|---|
| Self-reported confidence | intent correctness | AUC **0.539** — no usable signal |
| Evidence similarity | intent correctness | AUC 0.378 — none (wrong hypothesis, see below) |
| Evidence similarity | **reply grounding** | ρ = **+0.283**, p = 0.028 — modest but real |

The self-reported confidence the model emits is worthless for predicting its
own correctness, which retroactively justifies demoting it from primary
trigger to weak last check. Evidence similarity does not predict intent
accuracy — but it was never designed to; its claim is about whether a reply
can be grounded, and against *that* it holds up with a small, significant
positive relationship (grounding 4.73 on weak evidence vs 4.93 on strong).

---

## 11. Failure analysis

Derived from all 24 misclassifications in `reports/predictions.csv`.

### 1. Missing conversational context (the largest single cause)

**Example:** *"it is a roku TV actually"* → true `other`, predicted
`device_app_issue` (similarity 0.94).
**Also:** *"After it did that 6 or 7 times, it came on. Thanks for your help"*
→ true `praise_chatter`, predicted `playback_error`.
**Expected:** recognise these are continuations that cannot be classified alone.
**Actual:** confident labels from fragments.
**Why:** 16 of 60 examples are mid-thread; the referent of "it" is in a turn
the agent never receives. Macro-F1 drops 0.673 → 0.190 on this slice.
**Type:** data/task-definition, not model.
**Fix:** pass the preceding turns, or restrict input to openers. Worth it —
this is the difference between 0.487 and 0.673 headline macro-F1.

### 2. `live_tv_sports_issue` vs `playback_error` (5 cases, the biggest
confusion cell)

**Example:** *"Playback failure every commercial break! This is not the time
to choke. #WorldSeries2017"* → true `live_tv_sports_issue`, predicted
`playback_error`.
**Also:** *"Hulu Live has been unwatchable for more than 24 hrs. Constant
buffering"* → same confusion.
**Why:** these messages genuinely contain both signals — a playback symptom
occurring in a live-TV context. The model weights symptom vocabulary
("buffering", "playback failure") over context vocabulary ("live", "game").
**Type:** taxonomy design, not model error. The two classes overlap by
construction and a human could defend either label.
**Fix:** either merge them, or define precedence explicitly ("if the content
is live, live wins"). Cheap, and would recover a meaningful share of errors.

### 3. `device_app_issue` misread as `account_access` (2 cases)

**Example:** *"Where do I find settings?? Not on computer & can't find it on
the TV screen."* → predicted `account_access`.
**Why:** "can't find / can't get to something" reads as an access problem. The
real distinction is *navigating a UI* vs *authenticating*, which the intent
definitions state but do not make salient.
**Type:** prompt/definition clarity.
**Fix:** sharpen the boundary line in the taxonomy. Very cheap.

### 4. Sarcasm and irony read literally

**Example:** *"Eww! Hulu's community managers have to work on Saturday nights!
God, I'm so sorry. Are you okay? Blink twice if you need rescued!"* → true
`praise_chatter` (a joke), predicted `general_complaint`.
**Why:** surface sentiment is negative; the intent is friendly banter.
`praise_chatter` scores F1 **0.00** (both its examples missed).
**Type:** model limitation.
**Fix:** few-shot examples of ironic messages. Low priority — rare, and
misrouting a joke to a human costs almost nothing.

### 5. Label ambiguity — some "errors" are arguably correct

**Example:** *"can't watch anything. What's up? 🤷🏾‍♂️"* → labelled
`general_complaint`, predicted `playback_error`. The prediction is defensible;
so is the label.
**Why:** the `general_complaint`-vs-specific boundary was flagged as fuzzy
when the taxonomy was designed, and it is producing disagreements now.
**Type:** evaluation, not model.
**Fix:** this is precisely what a second annotator would quantify. Without an
inter-annotator agreement number we cannot say how much of the 40% error rate
is real. **This is the strongest argument for limitation 3 in §12.**

**Cross-cutting observation:** retrieval similarity on these errors is high
(0.80–0.95). The system finds topically similar history and still
misclassifies — consistent with the calibration finding that similarity does
not predict intent correctness.

---

## 12. What is misleading about my headline number?

**0. The headline number is measured on the slice where the system does best.**
"Macro-F1 0.673" is openers only (n=44). Across all 60 examples it is
**0.487**. Both are defensible — openers are the task the system was built
for — but quoting 0.673 without saying which population it describes would be
the single most misleading thing in this report. Both appear in §10.

1. **n=60, and the confidence intervals are wide enough to swallow most
   differences.** The headline's own interval is [0.463, 0.792] — a spread of
   0.33. Any gap between systems smaller than roughly ±0.1 macro-F1 is not
   evidence of anything. The point estimates in §10 should never be quoted
   without their intervals.

1b. **The reply-quality headline does not survive its own significance test.**
   Agent 4.88 vs copy-nearest 4.77 looks like a win and is not one: paired
   95% CI [−0.10, +0.35]. Reporting "4.88/5 reply quality" as evidence the
   agent works would be quoting a number that does not distinguish it from a
   baseline with no language model in it at all.

2. **Below the assignment's 150–250 golden-set range.** Stated in §6 with
   consequences quantified. Every number here would move on a larger set, and
   the direction is unknown.

3. **One annotator, one pass, no measured human ceiling.** Nobody
   double-labelled anything, so there is no inter-annotator agreement number.
   That means we cannot separate "the agent was wrong" from "the label was
   arguable" — and on boundaries like `general_complaint` vs. a specific
   intent, plenty of labels are arguable.

4. **7 true escalations out of 60.** One flipped prediction moves escalation
   recall by ~14 percentage points. The escalation comparison in §10 is close
   to noise and should not be used to claim the agent beats the rule baseline.

5. **Exactly 1 billing example.** The single highest-cost decision in the
   entire system — money — is measured on n=1. We can make no claim about
   billing performance at all.

6. **Judge-derived reply scores have no per-item human anchor.** The judge
   recovers a constructed quality ordering well (rho = -0.83), but that is not
   the same as agreeing with a person on a real, ambiguous reply. It also
   shows a ceiling effect (real replies score a flat 5.00) and one dimension,
   `relevance`, that rates a reply containing an invented refund 4.68/5. The
   agent scores 5.00 on three of six dimensions -- sitting exactly where the
   judge has been shown not to discriminate.

7. **The person who built the system also wrote the labels and the ratings.**
   Blinding reduces this; it does not remove it.

8. **Historical Hulu replies are treated as good, but were never verified to
   have resolved anything.** No reliable resolution signal exists in the raw
   data. Grounding in history assumes history worked.

9. **Survivorship bias.** Only conversations that happened publicly on Twitter
   are visible. Customers who called, used in-app help, or gave up silently
   are absent from both the corpus and the evaluation.

10. **2017 data.** Hulu's product and support playbooks have changed. Nothing
    here validates the grounding corpus against present-day Hulu.

11. **The `natural` stratum is the only one that estimates production
    performance.** Any blended number across strata describes a population
    that does not exist. Agent macro-F1: 0.519 natural vs 0.316 rare_boost.

12. **An escalation trigger fires on a signal that does not predict what it
    claims.** `weak_evidence` fired 6 times. Evidence similarity correlates
    with reply grounding at rho=+0.283 -- real, but small enough that 6
    escalations on that basis is a decision resting on a weak signal.

13. **Zero fabricated claims were detected, which is not the same as zero
    fabrications.** The detector is a regex for a specific set of phrasings.
    `judge_validity.py` shows it catches deliberately injected fabrications,
    but a novel phrasing would pass. "0/60" measures the detector as much as
    the model.

---

## 13. What I would build with one more week

Ordered by what the evaluation above says is actually broken:

1. **Finish the golden set to 200** using the already-built sampler, including
   the `escalation_sensitive` stratum — this directly fixes limitations 1, 4
   and 5, which are the three biggest.
2. **Double-label 50 examples blind, a day apart**, for a human agreement
   ceiling. Every accuracy claim is uninterpretable without it.
3. **Collect per-item human ratings** (~21 blind ratings, tooling already
   built: `make rate`). This is the one assignment deliverable not met, and
   it is the cheapest of these to close.
4. **Replace the `relevance` judge dimension** — it rates a reply carrying an
   invented refund 4.68/5. Probe the ceiling effect with deliberately
   near-miss replies rather than obvious degradations.
5. **Pass conversation context** for mid-thread messages, converting failure
   mode 1 from "out of scope" into a solved case — worth 0.487 -> 0.673
   macro-F1 on the blended set.
6. **Re-test k=1 vs k=3 on a held-out dev set.** The ablation found k=1
   significantly better than the shipped k=3, but on the test set — so acting
   on it now would be tuning on the evaluation data. With more labelled
   examples, split off a dev set, confirm there, and ship the winner.
7. **Cost curve for escalation** — sweep the evidence threshold and plot
   automation rate against missed-escalation rate, so the operating point is
   chosen from an explicit cost ratio rather than a percentile.

---

## 14. Decision log

16 entries in [`decision_log.md`](decision_log.md), each with the alternative
considered and why it was rejected. Bugs hit during development and how they
were diagnosed: [`build_log.md`](build_log.md).
