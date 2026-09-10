# Hulu Support Agent — Report

## 1. Executive summary

An AI support agent for `hulu_support` that classifies an incoming customer
message into one of 10 intents, drafts a reply grounded in retrieved
historical Hulu resolutions, and decides auto-handle vs. escalate with a
stated reason.

Evaluated against **three baselines** on 200 examples — **60 hand-labelled,
carrying every headline number**, and 140 AI-assisted, reported separately and
never blended (§6).

**Headline: agent macro-F1 0.487** [0.330, 0.617] on the hand-labelled set,
against 0.356 for TF-IDF, 0.283 for keyword rules and 0.040 for majority-class.
On conversation openers — the task the system is actually built for — it
reaches 0.638.

**Four findings that argue against this system, which matter more than the
headline:**

1. **Retrieval earns its place; the generation layer on top of it does not.**
   Removing retrieval entirely costs 0.87 judge points (CI [+0.67, +1.08]).
   But generating a reply does not beat copying the nearest historical reply
   (−0.03, CI [−0.12, +0.09]) — and at n=200 the point estimate flipped
   *against* the agent. The value is in the retrieval, not the LLM around it.
2. **On mid-thread messages the agent is worse than TF-IDF** (0.190 vs 0.240),
   with a 75% error rate. Fluency without context is a liability.
3. **The model's self-reported confidence is worthless** — AUC 0.539 for
   predicting its own correctness. An earlier version gated escalation on it.
4. **The system is weakest where errors cost most** — macro-F1 0.472 on the
   escalation-sensitive stratum versus 0.591 on natural traffic.

The most important sections are §12 (why not to trust the headline) and §4
(two evaluation bugs found by attacking our own results — one of which
invalidated every reply-quality number produced before it).

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
   the same conversation thread* (233 rows). Two tests fail if it regresses.

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

**200 examples, of mixed provenance, and the distinction is load-bearing:**

| Provenance | n | Used for |
|---|---:|---|
| `human` | 60 | **All headline results.** Labelled by hand via `src/label_tui.py`, no model prediction ever displayed. |
| `ai_claude` | 140 | Secondary analysis only. Labelled with AI assistance under time pressure. |

The `labelled_by` column in `data/golden/golden_labelled.csv` records this per
row, and `scripts/metrics.py` prints the two blocks separately with the human
block marked as the headline.

**Why the split is enforced rather than blended.** Scoring an AI system against
AI-produced labels measures agreement between two models, not correctness. A
high number on the AI-labelled rows would partly reflect two systems reading
the same taxonomy the same way. The AI labels were also not produced by
qwen2.5 (the generator) or llama3.1 (the judge), which avoids direct
circularity — but "not directly circular" is a weaker property than
"independent ground truth", and only the human rows have the latter.

**This is below the assignment's request for 150-250 examples the candidate
hand-labelled.** 60 were hand-labelled; the remaining 140 were not, and are
labelled as such rather than presented as human work. The consequences of the
smaller human set are quantified rather than hidden:

- confidence intervals on every headline metric are wide (§10)
- only 7 of the 60 human-labelled examples are true escalations
- exactly 1 human-labelled `billing_charge` example, so the highest-cost
  intent is effectively unmeasured

**Sampling.** Two tagged strata in the human set: `natural` (40, plain random
— the only slice that estimates production performance) and `rare_boost` (20,
spread evenly across the 10 TF-IDF clusters so low-frequency intents have any
measurable per-class score). The 140-row extension adds
`escalation_sensitive` and `adversarial` strata. Every row keeps its stratum
tag; results are reported per stratum rather than blended.

**Labelling protocol (human rows).** One annotator, one pass, working from the
definitions in §5, with no model prediction ever shown — a displayed
suggestion anchors annotator judgment and would make the golden set a
measurement of the model instead of a check on it. Escalation was judged per
message rather than derived from intent, which is why 3 labels escalate
outside the always-escalate categories.

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

200 evaluation examples: **60 hand-labelled (headline)** and 140 AI-assisted
(secondary). Reproduce with `make reproduce`.

### Intent classification

| System | Human-labelled only (n=60) **HEADLINE** | All 200 (secondary) |
|---|---|---|
| Trivial (majority class) | 0.040 [0.026, 0.060] | 0.034 [0.026, 0.041] |
| Simple (keyword rules) | 0.283 [0.156, 0.370] | 0.362 [0.299, 0.420] |
| Simple (TF-IDF, silver) | 0.356 [0.207, 0.473] | 0.361 [0.296, 0.419] |
| **Agent (LLM)** | **0.487** [0.330, 0.617] | **0.600** [0.508, 0.669] |

Macro-F1 with bootstrap 95% CIs. The agent beats every baseline on both sets;
on the human-labelled headline the interval against TF-IDF overlaps slightly,
so that particular gap is suggestive rather than established.

By message type (all 200): **openers 0.638** [0.533, 0.706] — the defined
task — versus **mid-thread 0.190** [0.033, 0.336], where the agent is beaten
by TF-IDF (0.240). Fluency without context is a liability, not an asset.

Per-intent (all 200), the classes that matter most now have enough support to
measure: `account_access` F1 0.81 (n=17), `billing_charge` 0.76 (n=18). The
weakest are `general_complaint` 0.40 and `other` 0.40 — both the fuzzy
boundaries flagged when the taxonomy was designed.

### Reply quality (LLM judge, 1–5)

| System | grounding | correctness | relevance | safety | tone | overall |
|---|---:|---:|---:|---:|---:|---:|
| Trivial (constant apology) | 3.04 | 3.55 | 4.70 | 5.00 | 4.55 | 3.58 |
| Simple (copy nearest reply) | 4.88 | 4.88 | 4.88 | 4.92 | 4.94 | **4.89** |
| Agent (grounded generation) | 4.84 | 4.88 | 4.99 | 5.00 | 5.00 | **4.87** |

**Paired difference, agent vs copy-nearest: −0.03, 95% CI [−0.12, +0.09].**

At n=200 the point estimate has flipped slightly *against* the agent, and the
interval still contains zero. **Generating a reply is not shown to beat simply
copying the most similar historical reply.** Both crush the constant apology,
so retrieval is doing the work — the generation layer on top of it is not
earning its cost on this metric. This is the clearest negative result in the
report and it argues against the system's own complexity.

### Escalation (40 true escalations of 200)

| System | Recall | Precision | Missed escalations | Needless escalations |
|---|---:|---:|---:|---:|
| Trivial (always escalate) | 1.00 | 0.20 | 0.00 | 1.00 |
| Simple (sensitive intents) | 0.60 | 0.89 | 0.40 | 0.02 |
| **Agent** (intent+evidence+claims) | **0.78** | 0.44 | **0.23** | 0.25 |

With 40 positives instead of the 7 available at n=60, this comparison is now
worth reading. The agent catches **78% of cases needing a human versus 60%**
for the rule baseline, at the cost of escalating 25% of cases that did not
need it (rules: 2%). Given the stated asymmetry — a wrong automated answer
about someone's money costs far more than a needless handoff — that trade is
defensible, but it is a *choice*, not a free win, and the right operating
point needs a cost ratio we have not measured (§13).

### Robustness

JSON parse success 200/200 for both classification and reply generation.
Fabricated-claim detections: 0/200, with `judge_validity.py` confirming the
detector fires on deliberately injected fabrications.

### By stratum

| Stratum | n | Agent macro-F1 |
|---|---:|---:|
| adversarial | 35 | 0.719 |
| natural | 90 | 0.591 |
| escalation_sensitive | 35 | 0.472 |
| rare_boost | 40 | 0.443 |

`natural` is the only stratum that estimates production performance. Note the
system does *worse* on `escalation_sensitive` (0.472) than on average —
weakest exactly where errors are most expensive.

### Do the uncertainty signals work? (`scripts/calibration.py`)

| Signal | Tested against | Result |
|---|---|---|
| Self-reported confidence | intent correctness | AUC **0.539** — no usable signal |
| Evidence similarity | intent correctness | AUC 0.378 — none (wrong hypothesis) |
| Evidence similarity | **reply grounding** | ρ = **+0.283**, p = 0.028 — modest but real |

The confidence number the model emits is worthless for predicting its own
correctness, which retroactively justifies demoting it from primary escalation
trigger to weak last check. Evidence similarity does not predict intent
accuracy — but it never claimed to; its claim is about whether a reply can be
grounded, and against that it holds up.

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
measured on the same examples used to report every other number here;
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

From all 72 misclassifications in `reports/predictions.csv` (n=200). The same
five patterns found at n=60 held at n=200, with much stronger counts.

### 1. Missing conversational context — the sharpest single effect

**Error rate on mid-thread messages: 75% (12/16). On openers: 33% (60/184).**

**Example:** *"it is a roku TV actually"* → true `other`, predicted
`device_app_issue`. **Also:** *"After it did that 6 or 7 times, it came on.
Thanks for your help"* → true `praise_chatter`, predicted `playback_error`.
**Expected:** recognise these cannot be classified alone.
**Actual:** confident labels from fragments whose referent ("it") is in a turn
the agent never receives.
**Type:** task definition, not model. **Fix:** pass prior turns, or restrict
input to openers. Worth it — macro-F1 goes 0.190 → 0.638 across this boundary.
**Note:** this is the one failure where the *simplest* baseline wins (TF-IDF
0.240 vs agent 0.190), because a bag-of-words model has no strong opinion to
be confidently wrong with.

### 2. `live_tv_sports_issue` bleeding into two other classes — 15 errors

The single largest confusion cluster: **9** to `playback_error`, **6** to
`content_availability`.

**Example (→ playback_error):** *"live is so poor right now. Chopped playback,
not holding video quality."*
**Example (→ content_availability):** *"NBC in Cincinnati 'temporarily
unavailable'. Makes it difficult to watch Sunday Night Football."*
**Why:** these messages genuinely contain both signals — a playback symptom or
an availability symptom *occurring in a live context*. The model weights the
symptom vocabulary ("chopped playback", "unavailable") over the context
vocabulary ("live", "Sunday Night Football").
**Type:** taxonomy design. The classes overlap by construction and a human
could defend either label.
**Fix:** state precedence explicitly — "if the content is live, live wins" —
or merge the classes. Cheap, and would recover a meaningful share of 15 errors.

### 3. Device problems read as generic complaints — 4 errors

**Example:** *"Why is the app on [device] such outdated trash....?"* → true
`device_app_issue`, predicted `general_complaint`.
**Why:** the message is dominated by frustration, and the actionable detail
(which device, what is broken) is carried implicitly. The classifier follows
sentiment over content.
**Type:** prompt/definition clarity. **Fix:** sharpen the rule already stated
in §5 — if any concrete fault or device is named, it is not a general
complaint. Very cheap.

### 4. Sarcasm and irony read literally

**Example:** *"Eww! Hulu's community managers have to work on Saturday nights!
God, I'm so sorry. Are you okay? Blink twice if you need rescued!"* → true
`praise_chatter` (a joke), predicted `general_complaint`.
**Why:** surface sentiment is negative; the intent is friendly banter.
`praise_chatter` recall is 0.50 even at n=6.
**Type:** model limitation. **Fix:** few-shot ironic examples. Low priority —
rare, and misrouting a joke to a human costs almost nothing.

### 5. Label ambiguity — some "errors" are defensible readings

**Example:** *"can't watch anything. What's up? 🤷🏾‍♂️"* → labelled
`general_complaint`, predicted `playback_error`. Both are reasonable.
`general_complaint` scores the lowest F1 of any class (0.40), and it is the
class whose boundary was flagged as fuzzy when the taxonomy was written.
**Type:** evaluation, not model. **Fix:** a second annotator would quantify how
much of the 36% error rate is real disagreement versus label noise. Without an
inter-annotator agreement number we cannot separate them — **the strongest
argument for limitation 3 in §12.**

**Cross-cutting:** retrieval similarity on these errors is high (0.80–0.95).
The system finds topically similar history and still misclassifies, consistent
with the calibration finding that similarity does not predict intent
correctness.

## 12. What is misleading about my headline number?

**0. Which number is the headline, and on which population.** Agent macro-F1
is **0.487** on the 60 hand-labelled examples, **0.600** across all 200, and
**0.638** on conversation openers. All three are real; quoting the highest
without saying which population it describes would be the most misleading
thing available. The human-labelled 0.487 is the one this report treats as the
result, because it is the only slice with independent ground truth.

**1. 140 of 200 labels were not produced by a human.** They were AI-assisted
(§6), recorded in `labelled_by`, and excluded from every headline. But their
presence still shapes secondary numbers: the per-intent table, confusion
matrix, and escalation comparison all run on the full 200 because the human
subset is too small to support them. Measuring an AI system against
AI-produced labels reflects agreement between two models. Where those numbers
look good, some of that is two systems reading one taxonomy the same way.

**2. The human-labelled set is small and its intervals are wide.** n=60, and
the headline's interval is [0.330, 0.617] — a spread of 0.29. The agent-vs-
TF-IDF gap on that slice is not established, only suggestive.

**3. One annotator, one pass, no measured human ceiling.** Nothing was
double-labelled, so there is no inter-annotator agreement number and no way to
separate "the agent was wrong" from "the label was arguable". On boundaries
like `general_complaint` — lowest F1 of any class at 0.40 — many labels are
genuinely arguable.

**4. The reply-quality headline does not survive its own significance test.**
Agent 4.87 vs copy-nearest 4.89, paired CI [−0.12, +0.09]. At n=200 the point
estimate flipped *against* the agent. Reporting "4.87/5 reply quality" as
evidence the system works would quote a number that does not distinguish it
from a baseline containing no language model at all.

**5. Judge scores have no per-item human anchor.** The judge recovers a
constructed quality ordering well (ρ = −0.83) and catches injected
fabrications, but that is not agreement with a person on a real, ambiguous
reply. It also shows a ceiling effect — real replies score a flat 5.00 — and
one dimension, `relevance`, rates a reply containing an invented refund
4.68/5. The agent sits at 5.00 on three of six dimensions, exactly where the
judge has been shown not to discriminate.

**6. The system is weakest where errors are most expensive.** Macro-F1 on the
`escalation_sensitive` stratum is 0.472, below its 0.591 on `natural`. Money
and account cases are the ones it handles least well.

**7. The escalation comparison encodes a cost ratio we never measured.** The
agent's higher recall (0.78 vs 0.60) is bought with 25% needless escalations
versus the rule baseline's 2%. Whether that is a win depends entirely on the
relative cost of a wrong auto-reply versus a wasted human minute — a number
this project asserts qualitatively and never quantifies.

**8. An escalation trigger fires on a weak signal.** `weak_evidence` fired 24
times. Evidence similarity correlates with reply grounding at ρ = +0.283 —
real, but small for something driving a quarter of all escalations.

**9. Zero fabricated claims detected is not zero fabrications.** The detector
is a regex over specific phrasings. It catches deliberately injected cases,
but a novel phrasing passes. "0/200" measures the detector as much as the model.

**10. Historical Hulu replies are treated as good and were never verified to
have resolved anything.** No reliable resolution signal exists in the raw data.
Grounding in history assumes history worked.

**11. Survivorship bias.** Only conversations that happened publicly on Twitter
are visible. Customers who called, used in-app help, or gave up silently are
absent from the corpus and the evaluation.

**12. 2017 data.** Hulu's product and support playbooks have changed. Nothing
here validates the grounding corpus against present-day Hulu.

**13. The person who built the system also produced the labels and the
AI-assisted labels.** Blinding during human labelling reduces this; it does not
remove it.

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
