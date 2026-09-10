# Hulu Support Agent — Report

## 1. Executive summary

An AI support agent for `hulu_support` that classifies an incoming customer
message into one of 10 intents, drafts a reply grounded in retrieved
historical Hulu resolutions, and decides auto-handle vs. escalate with a
stated reason.

Evaluated against **three baselines** on **200 hand-labelled examples** — all
200, labelled personally, no model suggestion ever shown during labelling.

**Headline: agent macro-F1 0.222** [0.159, 0.277], accuracy 28.5%, against
0.206 for TF-IDF, 0.152 for keyword rules, 0.047 for majority-class.

**That headline is itself the most important finding in this report — read
§12 first.** This project's golden set started at 60 examples and reached
0.487 macro-F1 / 72.7% accuracy on conversation openers. Expanding it to a
properly stratified 200 — adding the billing/account and adversarial cases
the first draw missed almost entirely — collapsed that to 0.222 / 28.5%. The
system did not get worse. The first measurement was wrong, on real,
first-hand evidence produced by this exact project. That is the mandatory
"what is misleading about my headline number" question, answered by what
actually happened while building this, not hypothetically.

**Five findings that matter more than the top-line number:**

1. **The agent's edge over TF-IDF is not established.** 0.222 [0.159, 0.277]
   vs. 0.206 [0.147, 0.260] — the intervals overlap substantially. Only the
   trivial baseline is clearly worse than everything else.
2. **The escalation safety net fails when classification fails upstream.**
   Of 12 true `billing_charge` messages, 7 (58%) were misclassified into a
   different intent and never escalated at all — the sensitive-intent rule
   only protects messages the classifier correctly recognizes as sensitive.
3. **The model over-predicts `billing_charge` on money-adjacent vocabulary.**
   Precision is 0.12 — of every message the agent calls `billing_charge`,
   seven in eight are something else (a playback complaint that happens to
   mention a dollar amount, a plan question). This drives needless escalation.
4. **Generating a reply does not beat copying the nearest historical one**
   (paired diff −0.03, CI [−0.12, +0.09], contains zero). Removing retrieval
   entirely does cost real quality (+0.87 judge points, CI far from zero) —
   the value is in the retrieval, not the generation layer on top of it.
5. **The model's self-reported confidence is worthless** (AUC 0.531 for
   predicting its own correctness) — the exact signal an earlier version of
   this system gated escalation on.

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
*toward* escalation — but §11 shows that principle only works if the intent
feeding it is correct, and often it is not.

**What was NOT built, and why:**

- **Multi-turn dialogue.** The agent answers one incoming message. Mid-thread
  follow-ups ("yes, during ads") need prior turns to mean anything; rather
  than half-support them, they are excluded from the defined task and
  measured separately (§10, by message type).
- **Fine-tuning.** Retrieval satisfies the "grounded in how this brand
  historically resolved it" requirement directly, stays inspectable, and
  updates when the corpus does. Fine-tuning would bake 2017 policy into
  weights and remove the audit trail.
- **Any agent/RAG framework.** Retrieval is a cosine similarity search over a
  numpy array. Every step is readable and explainable.
- **A production safety layer** beyond the escalation policy and the
  fabricated-claim regex. Real deployment would need PII handling and abuse
  detection.
- **Judge rubric tuning against results.** The rubric was written once from
  the failure modes we cared about, then validated (§9) independently of any
  golden-set score, so it could not be tuned to flatter this system.

---

## 3. Architecture

```
incoming message
      |
      +--> classify ------------> intent (10 classes + "other")
      |                           + self-reported confidence  [shown worthless, §10]
      |
      +--> retrieve ------------> 3 nearest historical Hulu exchanges
      |                           + top-1 similarity          [weak but real signal, §10]
      |
      +--> draft ---------------> reply constrained to that evidence
      |                           + fabricated-claim check
      |
      +--> escalate? -----------> auto-handle | escalate + reason code
```

Escalation fires on the first matching rule: sensitive intent
(billing/account) → unclassifiable (`other`) → weak evidence (top-1
similarity < 0.80) → low self-report. Separately, a fabricated claim detected
in the drafted reply **overrides** whatever that chain decided. The chain's
first rule is only as reliable as intent classification itself — see the
billing_charge finding in §11, which is the sharpest limitation of this
design.

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
   **Fixed:** escalation now gates primarily on measured retrieval evidence
   quality, with the threshold derived from the *reference corpus*
   (p10 of top-1 similarity = 0.80), not from the golden set.
   `scripts/calibration.py` tests whether either signal predicts correctness
   rather than assuming it — self-report still does not (§10).

**Other integrity measures.** All 200 golden labels made by one human, by
hand, with no model suggestion ever displayed during labelling — a shown
suggestion anchors annotator judgment and would make the set a measurement of
the model rather than a check on it. Silver (LLM) labels exist only to train
the TF-IDF baseline and are never scored against. Thresholds derived from
reference data, not the golden set. `temperature=0`, fixed seeds, every model
call cached to disk.

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

**200 examples, all hand-labelled by one person (me), in two sessions, via
`src/label_tui.py`.** No model prediction was ever shown during labelling —
a displayed suggestion would anchor the annotator's judgment to the system
being evaluated, which is exactly the independence the golden set exists to
provide.

**Sampling: four tagged strata, each answering a different question.**

| Stratum | n | Why |
|---|---:|---|
| `natural` | 90 | Plain random draw. The only slice that estimates real traffic. |
| `rare_boost` | 40 | Spread evenly across the 10 TF-IDF clusters, so low-frequency intents have any measurable per-class score. |
| `escalation_sensitive` | 35 | Messages matching billing/account/refund/security vocabulary. The first 60-example draw produced exactly **one** `billing_charge` example — the highest-cost intent in the whole system was unmeasured. |
| `adversarial` | 35 | Sarcasm, shouting, multi-question, very short/long. Built to find where the system breaks, not to estimate average quality. |

Every row keeps its stratum tag; results are reported per stratum (§10) and
never silently blended.

**Only conversation openers were targeted by the sampler**, but 16 of the 200
turned out to be mid-thread fragments in the underlying corpus (a message
that is itself a reply, but where the *customer's* turn reads as an opener in
isolation). Both are reported (§10) rather than mixed into one number.

**Labelling protocol.** Definitions from §5, no model prediction shown,
escalation judged per-message rather than mechanically derived from intent —
40 of 200 human escalation calls fall outside the two always-escalate
categories, reflecting real judgment about risk on a case-by-case basis.

---

## 7. Evaluation method

| Metric | Definition |
|---|---|
| **Macro-F1** (primary) | Unweighted mean of per-class F1. Primary because accuracy is carried by common intents while the ones that matter most (`billing_charge`, `account_access`) are rarest. |
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
receives. That biases the comparison against our own system deliberately —
and even so, §10 shows the gap over it is not statistically established.

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

**Spearman rho between the known ordering and the judge's score: −0.829
(p < 0.0001, n = 100). Pairwise ordering accuracy: 113/150 = 75%.**

All three discrimination checks pass. The judge ranks a genuine human reply
above a generic non-answer, notices the *subtle* case where a reply keeps its
tone but loses its actionable content (5.00 → 4.48), and drives safety to the
floor (1.00) on an injected fabrication.

**Three honest problems this same table exposes:**

1. **`relevance` barely works.** Spread across four wildly different reply
   qualities is 0.48 — it even rates the fabricated reply 4.68/5.
2. **Ceiling effect on good replies.** Real human replies score a flat 5.00 on
   every dimension. The judge separates good from bad but may not discriminate
   *among* good replies — consistent with the agent-vs-copy-nearest comparison
   in §10 failing its own significance test.
3. **These variants differ obviously**; real replies from two systems differ
   subtly. Recovering a constructed ordering does not establish agreement
   with a person on a real, ambiguous reply.

**Practical consequence, applied throughout §10:** judge scores are treated as
a ranking signal across systems, never as an absolute quality level.

---

## 10. Results

n = 200 hand-labelled examples. Reproduce with `make reproduce`.

### Intent classification

| System | macro-F1 | 95% CI | accuracy |
|---|---:|---|---:|
| Trivial (majority class) | 0.047 | [0.039, 0.054] | 0.305 |
| Simple (keyword rules) | 0.152 | [0.101, 0.198] | 0.200 |
| Simple (TF-IDF, silver-trained) | 0.206 | [0.147, 0.260] | 0.255 |
| **Agent (LLM)** | **0.222** | [0.159, 0.277] | 0.285 |

**Only the trivial baseline is clearly separated from the rest.** Rules,
TF-IDF and the agent all overlap each other's intervals — the agent's win is
a point estimate, not an established result, at this sample size.

**By message type:** openers (n=184) 0.220 [0.152, 0.276], accuracy 0.288;
mid-thread (n=16) 0.190 [0.033, 0.336], accuracy 0.250 — where TF-IDF (0.240)
beats the agent, because a bag-of-words model has no strong opinion to be
confidently wrong with when context is missing.

**Per-intent (agent):**

| Intent | Precision | Recall | F1 | n |
|---|---:|---:|---:|---:|
| content_availability | 0.52 | 0.41 | 0.46 | 61 |
| live_tv_sports_issue | 0.50 | 0.26 | 0.34 | 31 |
| other | 0.25 | 0.40 | 0.31 | 5 |
| playback_error | 0.26 | 0.26 | 0.26 | 34 |
| device_app_issue | 0.16 | 0.38 | 0.22 | 8 |
| account_access | 0.20 | 0.18 | 0.19 | 17 |
| billing_charge | **0.12** | 0.25 | 0.17 | 12 |
| feature_request | 0.12 | 0.18 | 0.14 | 11 |
| general_complaint | 0.14 | 0.12 | 0.13 | 17 |
| praise_chatter | 0.00 | 0.00 | 0.00 | 4 |

`billing_charge` precision of 0.12 means: of every message the agent labels
`billing_charge`, 7 in 8 are something else. See §11 for what that does to
escalation safety.

### Escalation (110 true escalations of 200)

| System | Recall | Precision | Missed escalations | Needless escalations |
|---|---:|---:|---:|---:|
| Trivial (always escalate) | 1.00 | 0.55 | 0.00 | 1.00 |
| Simple (sensitive intents) | 0.16 | 0.67 | 0.84 | 0.10 |
| **Agent** (intent+evidence+claims) | 0.39 | 0.61 | 0.61 | 0.31 |

With 110 positives this comparison is statistically usable, unlike the n=7 the
original 60-example set provided. **The agent still misses 61% of cases that
should escalate.** Which rule fired: `auto_handle` 129, `sensitive_intent` 39,
`weak_evidence` 24, `unclassifiable` 8.

### Reply quality (LLM judge, 1–5)

| System | grounding | correctness | relevance | safety | tone | overall |
|---|---:|---:|---:|---:|---:|---:|
| Trivial (constant apology) | 3.04 | 3.55 | 4.70 | 5.00 | 4.55 | 3.58 |
| Simple (copy nearest reply) | 4.88 | 4.88 | 4.88 | 4.92 | 4.94 | **4.89** |
| Agent (grounded generation) | 4.84 | 4.88 | 4.99 | 5.00 | 5.00 | **4.87** |

**Paired difference, agent vs copy-nearest: −0.03, 95% CI [−0.12, +0.09].**
Contains zero. Generating a reply is not shown to beat copying the most
similar historical one. Both crush the constant apology — retrieval is doing
the work; the generation layer on top of it is not earning its cost here.

### Robustness

JSON parse success 200/200 for both classification and reply generation.
Fabricated-claim detections: 0/200 — `judge_validity.py` confirms the
detector fires on deliberately injected fabrications, so this reads as a real
(if narrow-detector-limited) zero, not a parsing failure.

### By stratum

| Stratum | n | Agent macro-F1 | accuracy |
|---|---:|---:|---:|
| natural | 90 | 0.289 | 0.356 |
| rare_boost | 40 | 0.205 | 0.375 |
| adversarial | 35 | 0.112 | 0.200 |
| escalation_sensitive | 35 | **0.042** | 0.086 |

`natural` is the only stratum that estimates production performance —
**macro-F1 0.289, accuracy 35.6%, not the 72.7% the smaller first sample
suggested.** The system is worst by a wide margin exactly where the strata
were built to probe: money and account cases.

### Do the uncertainty signals work? (`scripts/calibration.py`)

| Signal | Tested against | Result |
|---|---|---|
| Self-reported confidence | intent correctness | AUC **0.531** — no usable signal |
| Evidence similarity | intent correctness | AUC 0.489 — none (wrong hypothesis, see below) |
| Evidence similarity | **reply grounding** | ρ = **+0.199**, p = 0.005 — real but weak |

The model's self-reported confidence carries essentially no information about
whether it is right (correct-case mean 0.979, wrong-case mean 0.971) —
confirming the decision to demote it from primary escalation trigger to a
last, weak check. Evidence similarity was never meant to predict intent
correctness; tested against what it actually claims (reply grounding), the
relationship is statistically real (p=0.005) but small — grounding averages
4.68 on weak evidence vs. 4.89 on strong, explaining under 4% of the variance.
Weak evidence alone is not a strong basis for the quarter of all escalations
it currently drives (`weak_evidence`: 24 of 200).

### Does retrieval earn its place? (`scripts/ablate_retrieval.py`)

Run on a 60-message subsample (the original hand-labelled set) rather than
all 200, for cost — this is disclosed, not a full-sample result. Same
messages, same judge, three configurations:

| Config | Judge overall | Paired diff vs shipped k=3 | |
|---|---:|---|---|
| k=0 — no evidence at all | 4.017 | +0.867 [+0.667, +1.083] | significant |
| k=1 — one historical exchange | **4.983** | −0.100 [−0.183, −0.017] | significant |
| k=3 — shipped configuration | 4.883 | — | |

**Retrieval is worth +0.87 over an ungrounded prompt** — the one central
architectural claim that survives its own test, and it sharpens the §10
finding that the value sits in retrieval, not generation.

**k=1 significantly beats the shipped k=3** — adding the 2nd and 3rd matches
made replies slightly worse, consistent with dilution. **Not acted on**,
because this was measured on the same examples used for every other number
here; changing the configuration on that basis would be test-set tuning.
Recorded as a validated hypothesis for a proper dev set (§13).

---

## 11. Failure analysis

From all 143 misclassifications in `reports/predictions.csv` (n=200,
accuracy 28.5%).

### 1. The escalation safety net fails when classification fails first — the most consequential finding in this report

**7 of 12 true `billing_charge` messages (58%) were never escalated**, because
the classifier assigned them a different intent first and the sensitive-intent
rule only fires on the intent it is given:

| Message | Predicted | Escalated? |
|---|---|---|
| "why can't I manage my devices when I'm billed thru iTunes" | `device_app_issue` | No |
| "Ummm I'm trying to put a gift card... Page not found" | `content_availability` | No |
| "Aye yo why on earth do you have this many ads?" | `feature_request` | No |
| "saying video no longer available... Why am I paying $40" | `playback_error` | No |
| "why can't I manage my devices..." | `device_app_issue` | No |
| "ABC isn't showing... Please advise" | `live_tv_sports_issue` | No |
| "hi trying to confirm Hulu live TV is available on..." | `content_availability` | No |

**Expected:** a real billing dispute always reaches a human. **Actual:** the
system silently auto-handles more than half of them, drafting a generic
troubleshooting reply for what is actually a money problem. **Why:** the
escalation policy's strongest rule is downstream of the classifier's weakest
category (`billing_charge` F1 0.17, precision 0.12). **Type:** architecture,
not a training-data problem — a rule that trusts an upstream signal it has
already shown to be unreliable. **Fix:** a second, independent check for
money/account keywords at the escalation stage itself, not routed through
intent classification at all — closing the loop that `weak_evidence` and
`unclassifiable` almost, but do not, cover for this specific case.

### 2. `billing_charge` is also over-triggered by surface vocabulary — the mirror image of finding 1

Precision 0.12: of every message the agent calls `billing_charge`, 7 in 8 are
not. **Example:** a `playback_error` message mentioning a dollar figure
("why am I paying $40 for this bullshit") gets pulled toward `billing_charge`
by the presence of money-adjacent words, not by the actual complaint. **Type:**
model limitation — the classifier is pattern-matching vocabulary, not
intent. **Fix:** few-shot examples explicitly contrasting "mentions money" with
"is a billing dispute" in the classification prompt.

### 3. Missing conversational context

**Error rate on mid-thread messages: 75% (12/16). On openers: 32% (59/184).**
**Example:** *"it is a roku TV actually"* → true `other`, predicted
`device_app_issue`; its referent ("it") is in a turn the agent never receives.
**Type:** task definition. **Fix:** pass prior turns, or restrict input to
openers — worth 0.190 → 0.220 macro-F1 across this boundary alone.

### 4. `live_tv_sports_issue` bleeding into `playback_error` and `content_availability`

The largest topical confusion cluster (15 of 31 true `live_tv_sports_issue`
cases misrouted). **Example:** *"live is so poor right now. Chopped
playback"* → predicted `playback_error`. **Why:** the message genuinely
contains both a playback symptom and a live-context cue, and the model weighs
the symptom vocabulary more heavily. **Type:** taxonomy design — the classes
overlap by construction. **Fix:** state precedence explicitly ("if the
content is live, live wins") or merge the classes.

### 5. Label ambiguity on the fuzziest boundary

`general_complaint` is the weakest class (F1 0.13). **Example:** *"can't
watch anything. What's up?"* — labelled `general_complaint`, predicted
`playback_error`; both readings are defensible. **Type:** evaluation, not
model. **Fix:** a second annotator, blind, would quantify how much of the
71.5% error rate is real model failure versus genuinely arguable labels —
**the strongest single argument for limitation 3 in §12**, since it is
currently unmeasured.

---

## 12. What is misleading about my headline number?

**0. The headline number changed by 3× during this project, and that change
is the finding, not an embarrassment to hide.** The original 60-example
golden set (drawn before the `escalation_sensitive` and `adversarial` strata
existed) gave macro-F1 0.487, 0.673 on openers, 72.7% accuracy. Expanding to a
properly stratified 200 — which surfaced exactly the billing/account cases the
first draw almost entirely missed (1 billing example out of 60) — collapsed
this to macro-F1 0.222, accuracy 28.5%. Nothing about the system changed
between these two measurements. The sample did. **If this report quoted only
the first number, it would be presenting a real, reproducible artifact of
small-sample luck as a result.** This is not a hypothetical caution; it is
what happened, with git history and both prediction files to show it.

**1. Which population any single number describes.** Macro-F1 is 0.222 across
all 200, 0.220 on openers, 0.289 on the `natural` stratum alone (the one that
estimates production traffic), and 0.042 on `escalation_sensitive`. All four
are correct and describe different things. Quoting the highest without saying
which population it covers would be the most misleading move available.

**2. One annotator, one pass, no measured human ceiling.** Nothing was
double-labelled, so there is no inter-annotator agreement number and no way to
separate "the agent was wrong" from "the label was arguable" — most visible on
`general_complaint`, the lowest-F1 class, where several errors are defensible
readings (§11.5).

**3. The reply-quality headline does not survive its own significance test.**
Agent 4.87 vs. copy-nearest 4.89, paired CI [−0.12, +0.09]. Reporting "4.87/5"
as evidence the generation step works would quote a number that does not
distinguish it from a baseline with no language model in it.

**4. Judge scores have no per-item human anchor.** The judge recovers a
constructed quality ordering well (ρ = −0.83, §9) and catches injected
fabrications, but that is not the same as agreeing with a person on a real,
ambiguous reply. It also shows a ceiling effect — real replies score a flat
5.00 — exactly where the agent-vs-baseline comparison needed it to
discriminate and, per finding above, could not.

**5. The escalation policy's headline recall (39%) undersells how it fails.**
It is not uniformly weak — it is *structurally* weak on exactly the category
where failure costs most, because the rule trusts an intent classifier that is
worst on that same category (§11.1). A single aggregate recall number hides
that the failure is concentrated, not spread evenly.

**6. Zero fabricated claims detected is not zero fabrications.** The detector
is a regex over specific phrasings; it catches deliberately injected cases
(§9) but a novel phrasing would pass. "0/200" measures the detector as much as
the model.

**7. The retrieval ablation used a 60-message subsample, not all 200,** for
cost reasons, disclosed in §10. The two numbers it produced (retrieval matters;
k=1 beats k=3) are real on that subsample but have not been confirmed at n=200.

**8. Historical Hulu replies are assumed to represent good resolutions and
were never verified to have actually resolved anything** — no reliable
resolution signal exists in the raw Twitter data.

**9. Survivorship bias.** Only conversations that happened publicly on
Twitter are visible in this corpus or this evaluation; customers who called,
used in-app help, or gave up silently are invisible to both.

**10. This is 2017 Twitter data.** Hulu's product and support playbooks have
changed since; nothing here validates the grounding corpus against present-day
Hulu.

**11. The person who built the system also produced every label.** Blinding
during labelling (no model suggestion shown) reduces this; it does not
remove it, and is exactly why limitation 2 matters as much as it does.

---

## 13. What I would build with one more week

Ordered by what the evaluation above shows is actually broken, not by what
would look most sophisticated:

1. **Fix the escalation-classification coupling directly (§11.1).** Add an
   independent keyword/embedding check for money and account risk that fires
   at the escalation stage regardless of what intent was assigned. This is
   the single highest-value fix identified — it closes a failure mode that
   currently lets the majority of real billing disputes through unescalated.
2. **Double-label 50 examples blind, a day apart**, for a human agreement
   ceiling. Every accuracy claim in this report is currently uninterpretable
   without one, especially on `general_complaint`.
3. **Collect per-item human ratings** (~21 blind ratings, tooling already
   built: `make rate`). The one assignment deliverable not met, and the
   cheapest of these to close.
4. **Re-run the retrieval ablation on all 200 examples**, not the 60-message
   subsample it currently uses, to confirm the k=1-beats-k=3 finding holds at
   the full sample size before ever acting on it.
5. **Replace the `relevance` judge dimension** — it rates a reply carrying an
   invented refund 4.68/5 and contributes almost no discriminative signal.
6. **Pass conversation context** for mid-thread messages, converting failure
   mode 3 (§11) from "out of scope" into a solved case.
7. **Re-test k=1 vs. k=3 on a genuine held-out dev set** before shipping a
   change — the ablation result is currently test-set-tuned information.
8. **Cost curve for escalation** — sweep the evidence threshold and plot
   automation rate against missed-escalation rate, so the operating point is
   chosen from an explicit, stated cost ratio.

---

## 14. Decision log

Non-obvious decisions in [`decision_log.md`](decision_log.md), each with the
alternative considered and why it was rejected. Bugs hit during development
and how they were diagnosed: [`build_log.md`](build_log.md).
