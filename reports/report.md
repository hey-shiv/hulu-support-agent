# Hulu Support Agent — Report

## 1. Executive summary

An AI support agent for `hulu_support` that classifies an incoming customer
message into one of 10 intents, drafts a reply grounded in retrieved
historical Hulu resolutions, and decides auto-handle vs. escalate with a
stated reason.

Evaluated against **three baselines** on **200 hand-labelled examples**, all
labelled personally, no model suggestion ever shown during labelling.

**Headline: agent macro-F1 0.606** [0.513, 0.673], accuracy 63.0%, against
0.369 for keyword rules, 0.355 for TF-IDF, 0.035 for majority-class. The
agent's lead over both simple baselines is statistically established —
intervals do not overlap.

**This number went through three versions, and the middle one is the most
instructive part of this report.** A first 60-example draw gave macro-F1
0.487. Expanding to a stratified 200 initially *collapsed* the result to
0.222 — before an audit found the real cause was a labelling-session data
error, not a model weakness (§4). Correcting it gives the 0.606 reported here,
now backed by adequate samples on every intent, including the two that matter
most (`billing_charge` n=18, `account_access` n=17). §4 is this project's
real, documented answer to "what is misleading about my headline number,"
not a hypothetical.

**Findings that still matter, on the corrected data:**

1. **Reply generation does not beat copying the nearest historical reply**
   (paired diff −0.03, CI [−0.12, +0.09], contains zero). Removing retrieval
   does cost real quality (+0.87 judge points, CI far from zero) — the value
   is in retrieval, not the generation layer on top of it.
2. **`live_tv_sports_issue` is the sharpest remaining confusion**: 15 of 30
   true cases misrouted, mostly to `playback_error`, because messages
   genuinely carry both a playback symptom and a live-context cue and the
   model weighs the symptom vocabulary more heavily.
3. **The model's self-reported confidence is a weak signal at best** (AUC
   0.586) — better than it looked on the corrupted data (0.531-0.539) but
   still not something to gate a high-stakes decision on alone.
4. **Escalation recall is 54%**, missing nearly half of cases that should go
   to a human — though on the category that matters most, `billing_charge`,
   only 2 of 18 (11%) are missed, because the classifier is now genuinely
   strong there (F1 0.76).

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
  measured separately (§10, by message type).
- **Fine-tuning.** Retrieval satisfies the "grounded in how this brand
  historically resolved it" requirement directly, stays inspectable, and
  updates when the corpus does.
- **Any agent/RAG framework.** Retrieval is a cosine similarity search over a
  numpy array. Every step is readable and explainable.
- **A production safety layer** beyond the escalation policy and the
  fabricated-claim regex.
- **Judge rubric tuning against results.** The rubric was validated (§9)
  independently of any golden-set score, so it could not be tuned to flatter
  this system.

---

## 3. Architecture

```
incoming message
      |
      +--> classify ------------> intent (10 classes + "other")
      |                           + self-reported confidence  [weak signal, §10]
      |
      +--> retrieve ------------> 3 nearest historical Hulu exchanges
      |                           + top-1 similarity          [weak signal, §10]
      |
      +--> draft ---------------> reply constrained to that evidence
      |                           + fabricated-claim check
      |
      +--> escalate? -----------> auto-handle | escalate + reason code
```

Escalation fires on the first matching rule: sensitive intent
(billing/account) → unclassifiable (`other`) → weak evidence (top-1
similarity < 0.80) → low self-report. A fabricated claim detected in the
drafted reply **overrides** whatever that chain decided.

---

## 4. Data and evaluation integrity

**Source.** Kaggle `thoughtvector/customer-support-on-twitter`, 2.8M tweets.
Reconstructed into 1.26M `(customer message → brand reply)` exchanges by
walking backward from each reply's parent pointer.

**Brand: `hulu_support`.** Chosen on measured deflection rate:

| Brand | Exchanges | Deflection rate |
|---|---:|---:|
| TMobileHelp | 34,215 | 82.4% |
| AppleSupport | 106,646 | 55.7% |
| Uber_Support | 56,160 | 45.4% |
| AmazonHelp | 168,814 | 10.8% |
| **hulu_support** | **21,681** | **7.4%** |

**A retrieval leakage bug**, found and fixed early: the index originally
contained all 21,681 exchanges including every golden example; a test message
retrieved *itself* at similarity `1.0000`. Fixed by excluding every golden
example and its whole conversation thread (233 rows). Two tests fail if this
regresses.

**The labelling data-quality bug — the most consequential thing found in this
project.** A second labelling pass over 140 new examples, done quickly under
time pressure, produced macro-F1 0.222 — implausibly low against the first
batch's 0.487. Re-reading the `billing_charge` labels against their text found
`billing_charge` assigned to *"Are the new Christmas Movies going to be on?"*
and similarly unrelated messages. Auditing all 140 the same way found **101
of 140 (72%) mismatched to their content** — the pattern a drifting label
session produces, concentrated in `playback_error` (used as a de facto
catch-all) and `content_availability`.

Every correction was made by re-reading the message against the §5
definitions, the same standard used for the original 60. Both versions of the
golden set are preserved in git history — this is auditable, not asserted.
All results in this report use the corrected 200; the 0.222 figure measured
labelling noise, not the system, and would have shipped as a false finding of
badly-overestimated performance if not caught.

**Other integrity measures.** All 200 golden labels made by one human, by
hand, no model suggestion ever shown during labelling. Silver (LLM) labels
train the TF-IDF baseline only and are never scored against. Thresholds
derived from reference data, not the golden set. `temperature=0`, fixed
seeds, every model call cached.

---

## 5. Intent taxonomy

Derived by clustering 14,080 conversation openers (TF-IDF + KMeans), then
collapsed by hand — the clustering put 51% of messages in one shapeless
cluster and another was simply the show name "Rick and Morty."

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

**200 examples, all hand-labelled by one person, in two sessions**, via
`src/label_tui.py`. No model prediction was ever shown during labelling. The
second session's labels were subsequently audited and corrected as described
in §4.

**Sampling: four tagged strata, each answering a different question.**

| Stratum | n | Why |
|---|---:|---|
| `natural` | 90 | Plain random draw. The only slice that estimates real traffic. |
| `rare_boost` | 40 | Spread evenly across the 10 TF-IDF clusters, so low-frequency intents have any measurable per-class score. |
| `escalation_sensitive` | 35 | Messages matching billing/account/refund/security vocabulary. The first 60-example draw produced exactly **one** `billing_charge` example. |
| `adversarial` | 35 | Sarcasm, shouting, multi-question, very short/long. Built to find where the system breaks. |

Every row keeps its stratum tag; results are reported per stratum (§10).

**Only conversation openers were targeted**, but 16 of 200 turned out to be
mid-thread fragments in the underlying corpus. Both are reported separately
(§10).

**Labelling protocol.** Definitions from §5, no model prediction shown,
escalation judged per-message rather than mechanically derived from intent.

---

## 7. Evaluation method

| Metric | Definition |
|---|---|
| **Macro-F1** (primary) | Unweighted mean of per-class F1. Primary because accuracy is carried by common intents while the ones that matter most are rarest. |
| Accuracy | Reported alongside, never alone. |
| Per-class P/R/F1 + confusion matrix | Where the aggregate hides failure. |
| **Missed escalation rate** | Of messages that should escalate, the fraction auto-handled. |
| **Needless escalation rate** | Of messages that need not escalate, the fraction escalated. |
| Judge dimensions | grounding, correctness, relevance, safety, tone, overall (1–5). |
| Bootstrap 95% CI | 2000 resamples with replacement. |

---

## 8. Baselines

| | What it is | Why included |
|---|---|---|
| **Trivial** | Always predict the majority intent; one fixed apology; always escalate | The floor. |
| **Simple (rules)** | Hand-written keyword classifier; copy the nearest historical reply; escalate on sensitive intents | Independent of the LLM — an afternoon's engineering. |
| **Simple (TF-IDF)** | TF-IDF + logistic regression on 400 silver-labelled messages | Circular by construction — distills the LLM's own labels, so it cannot meaningfully exceed it. |

---

## 9. Is the judge trustworthy?

Reply-quality numbers come from an LLM judge (llama3.1) scoring replies from a
different model family (qwen2.5) — helpful, but not proof, so it was tested
directly. **Per-item human ratings were not collected** — a real, stated gap;
`scripts/judge_agreement.py` says so plainly rather than substituting weaker
evidence silently.

**Graded degradation test instead** (`scripts/judge_validity.py`, 25
exchanges × 4 constructed-quality variants = 100 judge calls):

| Variant | grounding | correctness | relevance | safety | tone | overall |
|---|---:|---:|---:|---:|---:|---:|
| real Hulu reply | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | **5.00** |
| same reply, actionable step removed | 4.36 | 4.36 | 4.52 | 4.84 | 4.68 | **4.48** |
| generic apology | 2.68 | 3.32 | 4.84 | 5.00 | 4.48 | **3.32** |
| real reply + invented refund/timeline | 1.92 | 1.56 | 4.68 | **1.00** | 3.68 | **2.16** |

**Spearman rho between the known ordering and judge score: −0.829
(p < 0.0001). Pairwise ordering accuracy: 75%.** The judge ranks a real reply
above a non-answer, catches the subtle loss of actionable content (5.00 →
4.48), and drives safety to the floor on an injected fabrication.

**Honest problems the same table exposes:** `relevance` barely
discriminates (spread 0.48, rates the fabricated reply 4.68/5); real replies
hit a ceiling of 5.00, so the judge may not separate good from *slightly
better*; and constructed variants differ obviously in a way real
system-vs-system replies do not.

**Practical consequence:** judge scores are used as a ranking signal, never
as an absolute quality level.

---

## 10. Results

n = 200 hand-labelled examples, corrected as described in §4. Reproduce with
`make reproduce`.

### Intent classification

| System | macro-F1 | 95% CI | accuracy |
|---|---:|---|---:|
| Trivial (majority class) | 0.035 | [0.027, 0.042] | 0.210 |
| Simple (keyword rules) | 0.369 | [0.301, 0.425] | 0.390 |
| Simple (TF-IDF, silver-trained) | 0.355 | [0.288, 0.409] | 0.440 |
| **Agent (LLM)** | **0.606** | [0.513, 0.673] | 0.630 |

No interval overlap — the agent's lead over both simple baselines is
statistically established.

**By message type:** openers (n=184) 0.645 [0.555, 0.710]; mid-thread (n=16)
0.190 [0.033, 0.336], where TF-IDF (0.240) beats the agent — a bag-of-words
model has no strong opinion to be confidently wrong with when context is
missing.

**Per-intent (agent):**

| Intent | Precision | Recall | F1 | n |
|---|---:|---:|---:|---:|
| account_access | 0.87 | 0.76 | 0.81 | 17 |
| praise_chatter | 1.00 | 0.57 | 0.73 | 7 |
| billing_charge | 0.67 | 0.89 | 0.76 | 18 |
| content_availability | 0.71 | 0.81 | 0.76 | 42 |
| live_tv_sports_issue | 0.81 | 0.43 | 0.57 | 30 |
| playback_error | 0.49 | 0.68 | 0.57 | 25 |
| device_app_issue | 0.63 | 0.48 | 0.55 | 25 |
| feature_request | 0.47 | 0.50 | 0.48 | 16 |
| general_complaint | 0.43 | 0.46 | 0.44 | 13 |
| other | 0.38 | 0.43 | 0.40 | 7 |

The two intents the escalation policy depends on most, `billing_charge` and
`account_access`, are now the two with the strongest recall (0.89, 0.76) —
directly consequential for §11.

### Escalation (72 true escalations of 200)

| System | Recall | Precision | Missed escalations | Needless escalations |
|---|---:|---:|---:|---:|
| Trivial (always escalate) | 1.00 | 0.36 | 0.00 | 1.00 |
| Simple (sensitive intents) | 0.35 | 0.93 | 0.65 | 0.02 |
| **Agent** (intent+evidence+claims) | 0.54 | 0.55 | 0.46 | 0.25 |

**On `billing_charge` specifically, only 2 of 18 (11%) are missed** —
the upstream classifier is now reliable there. Rules fired: `auto_handle` 129,
`sensitive_intent` 39, `weak_evidence` 24, `unclassifiable` 8.

### Reply quality (LLM judge, 1–5)

| System | grounding | correctness | relevance | safety | tone | overall |
|---|---:|---:|---:|---:|---:|---:|
| Trivial (constant apology) | 3.04 | 3.55 | 4.70 | 5.00 | 4.55 | 3.58 |
| Simple (copy nearest reply) | 4.88 | 4.88 | 4.88 | 4.92 | 4.94 | **4.89** |
| Agent (grounded generation) | 4.84 | 4.88 | 4.99 | 5.00 | 5.00 | **4.87** |

**Paired difference, agent vs copy-nearest: −0.03, 95% CI [−0.12, +0.09].**
Contains zero — this metric does not depend on intent labels, so it is
unchanged by the §4 correction. Generating a reply is not shown to beat
copying the most similar historical one; retrieval is doing the work.

### By stratum

| Stratum | n | Agent macro-F1 | accuracy |
|---|---:|---:|---:|
| adversarial | 35 | 0.719 | 0.771 |
| natural | 90 | 0.592 | 0.600 |
| escalation_sensitive | 35 | 0.469 | 0.657 |
| rare_boost | 40 | 0.423 | 0.550 |

`natural` is the only stratum estimating production performance — its 0.592
sits close to the overall headline, a reassuring sign the sampling is not
distorting the result.

### Do the uncertainty signals work? (`scripts/calibration.py`)

| Signal | Tested against | Result |
|---|---|---|
| Self-reported confidence | intent correctness | AUC **0.586** — weak signal |
| Evidence similarity | intent correctness | AUC 0.556 — weak (wrong hypothesis, see below) |
| Evidence similarity | **reply grounding** | ρ = **+0.199**, p = 0.005 — real but small |

Both signals are weak enough that neither should be the sole basis for an
escalation decision — consistent with the layered design in §3, where
sensitive-intent is checked first and these are secondary triggers only.

### Does retrieval earn its place? (`scripts/ablate_retrieval.py`)

60-message subsample, same messages, same judge, three configurations:

| Config | Judge overall | Paired diff vs shipped k=3 | |
|---|---:|---|---|
| k=0 — no evidence at all | 4.017 | +0.867 [+0.667, +1.083] | significant |
| k=1 — one historical exchange | **4.983** | −0.100 [−0.183, −0.017] | significant |
| k=3 — shipped configuration | 4.883 | — | |

Retrieval is worth +0.87 over an ungrounded prompt. k=1 significantly beats
the shipped k=3 — not acted on, since this was measured on the same examples
used for every other number here; changing configuration on that basis would
be test-set tuning (§13).

---

## 11. Failure analysis

74 misclassifications remain in the corrected 200 (accuracy 63%).

### 1. `live_tv_sports_issue` bleeding into `playback_error`/`content_availability` — the largest cluster

15 of 30 true cases misrouted (9, 6). **Example:** *"live is so poor right
now. Chopped playback"* → predicted `playback_error`. **Example:** *"NBC in
Cincinnati 'temporarily unavailable'... Sunday Night Football"* → predicted
`content_availability`. **Why:** these messages carry both a symptom and a
live-context cue, and the model weighs the symptom more heavily. **Type:**
taxonomy design — the classes overlap by construction. **Fix:** state
precedence ("if live, live wins") or merge the classes.

### 2. `content_availability` ↔ `feature_request`, symmetric confusion

4 errors each direction. **Example (→ feature_request):** a missing-title
message reads as ambiguous between "not there yet" and "please add this."
**Type:** genuine boundary ambiguity. **Fix:** an operational rule for
"existing content" vs. "new capability."

### 3. Missing conversational context

75% error rate on mid-thread fragments (12/16) vs. 34% on openers.
**Example:** *"it is a roku TV actually"* → true `other`, predicted
`device_app_issue`; its referent lives in a turn the agent never sees.
**Type:** task definition, not model. **Fix:** pass prior turns, or keep
restricting evaluation to openers.

### 4. Device problems read as playback errors

4 errors. **Example:** *"Update sucks and now we can't even watch"* → true
`device_app_issue`, predicted `playback_error`. **Why:** "can't watch" reads
as a stream symptom even when the fault is app-specific. **Type:** model
limitation. **Fix:** few-shot examples distinguishing stream faults from
app-itself faults.

### 5. `praise_chatter`: low recall, perfect precision

Recall 0.57 (4/7), precision 1.00 — when predicted it is always right, but it
misses several. **Example:** sarcastic messages ("Blink twice if you need
rescued!") read as complaints on the surface. **Type:** irony specifically.
**Fix:** low priority — misrouting a joke to a human costs almost nothing.

---

## 12. What is misleading about my headline number?

**0. The headline number is only trustworthy because a labelling error was
caught and fixed — that process is itself the finding.** Before correction,
this exact 200-example set gave macro-F1 0.222 — looking like sobering
evidence the system was weaker than a smaller sample suggested. It was not;
it was evidence that 72% of one labelling session's outputs were wrong.
**A report presenting one number without describing how its ground truth was
produced and checked is asking to be trusted on faith**; this one can point
to the specific, auditable correction instead (§4, §14).

**1. One annotator, no measured inter-annotator ceiling, even after
correction.** The fix caught content-label mismatches; it could not resolve
genuinely ambiguous boundaries like `general_complaint` vs. a specific
intent, which remain one person's judgment throughout.

**2. Which population a number describes.** Macro-F1 is 0.606 across all 200,
0.645 on openers (the defined task), 0.592 on the `natural` stratum alone.
All are correct and describe different things.

**3. The reply-quality headline does not survive its own significance test.**
Agent 4.87 vs. copy-nearest 4.89, paired CI [−0.12, +0.09]. This number is
untouched by the §4 correction (it depends on replies, not intent labels),
and it still does not support "the LLM generation step improves quality."

**4. Judge scores have no per-item human anchor** (§9) and show a ceiling
effect exactly where the agent-vs-baseline reply comparison needed
discrimination.

**5. `relevance` barely functions as a judge dimension** — it rates a reply
carrying an invented refund 4.68/5.

**6. Escalation recall (54%) still misses nearly half of true cases overall**,
even though the category costing most, `billing_charge`, is well-protected
(89% recall). A single aggregate number would hide that the remaining risk is
concentrated in lower-stakes categories.

**7. Zero fabricated claims detected is not zero fabrications** — the
detector is a regex over specific phrasings and a novel one would pass.

**8. The retrieval ablation used a 60-message subsample**, not all 200, for
cost — disclosed in §10, not yet confirmed at full sample size.

**9. Historical Hulu replies are assumed to represent good resolutions and
were never verified to have resolved anything.**

**10. Survivorship bias** — only public Twitter conversations are visible to
this corpus or this evaluation.

**11. This is 2017 data**; Hulu's product and support playbooks have changed.

**12. The person who built the system produced every label, including the
corrections** — blinding during labelling reduces this, it does not remove it.

---

## 13. What I would build with one more week

1. **Re-run the retrieval ablation on all 200 examples**, not the 60-message
   subsample, to confirm the k=1-beats-k=3 finding before ever acting on it.
2. **Double-label 50 examples blind, a day apart**, for a genuine
   inter-annotator agreement ceiling — the one thing the §4 correction could
   not provide, since it fixed factual mismatches, not judgment variance.
3. **Collect per-item human ratings** (`make rate`, ~4 minutes) — the one
   assignment deliverable not met.
4. **Fix the `live_tv_sports_issue` boundary** (§11.1) with explicit
   precedence rules — the single largest remaining error cluster.
5. **Replace the `relevance` judge dimension.**
6. **Pass conversation context** for mid-thread messages.
7. **Re-test k=1 vs. k=3 on a genuine held-out dev set** before shipping any
   change.
8. **Cost curve for escalation** — sweep the evidence threshold against an
   explicit, stated cost ratio rather than a percentile default.

---

## 14. Decision log

Non-obvious decisions in [`decision_log.md`](decision_log.md), including the
labelling-correction methodology as its own entry. Bugs hit during
development, including the full diagnosis of the labelling data-quality
issue: [`build_log.md`](build_log.md).
