# Decision log

Non-obvious decisions, the alternative considered, and why it was rejected.

---

**1. Brand: `hulu_support`, not any of the higher-volume accounts.**
*Reason:* Measured deflection rate -- how often a "reply" just redirects the
customer elsewhere ("DM us", "contact us here") instead of resolving anything.
TMobileHelp 82%, AppleSupport 56%, Uber 45%, AmazonHelp 11% but with much of
its volume being link-redirects too. hulu_support: 7.4%.
*Alternative:* AmazonHelp, 168k exchanges vs Hulu's 21.7k -- 8x the data.
*Rejected because:* a reply-drafting agent grounded in history is bounded by
what that history contains. Grounding on "please DM us" teaches the system to
deflect. 21.7k substantive exchanges beat 168k mostly-redirects.

**2. Reconstructed conversations by walking backward from brand replies.**
*Reason:* `in_response_to_tweet_id` lives on the reply and points at its
parent. Starting from replies means one indexed lookup per pair; every reply
by definition has a parent, so "no match" never needs handling.
*Alternative:* iterate customer messages and search for their replies.
*Rejected because:* nothing on a customer message points forward, so this
requires scanning the full 2.8M-row table per message, and most customer
messages have no reply at all.

**3. Ten intents, hand-built from clusters rather than taken from them.**
*Reason:* TF-IDF+KMeans over 21k messages put 51% of everything into one
shapeless cluster and made another cluster that was simply the show name
"Rick and Morty". Vocabulary clustering groups by shared words, not shared
meaning. The clusters informed the taxonomy; reading messages produced the
categories the clustering could not see (billing, account access, feature
requests).
*Alternative:* use the 10 clusters directly as the label set.
*Rejected because:* "Rick and Morty" is a topic, not an intent, and a
51%-of-everything class makes per-class metrics meaningless.

**4. Kept an explicit `other` class.**
*Reason:* Forces the system to have somewhere to put messages it genuinely
cannot categorise, and makes that visible in metrics instead of hidden inside
a wrong confident label. `other` also triggers escalation.
*Alternative:* nine substantive classes, forcing every message into one.
*Rejected because:* it converts "I don't know" into a confident wrong answer,
which is exactly the failure the escalation policy exists to prevent.

**5. Escalation is a cost-asymmetric decision with four distinct triggers, not
one boolean.**
*Reason:* Sensitive intent (money/account), unclassifiable intent, weak
retrieval evidence, and a detected fabricated claim are independent failure
routes and each names a different reason a human is needed. Every trigger
fails toward escalation, because a needless handoff costs one person's time
while a wrong automated answer about someone's money can cost far more.
*Alternative:* a single confidence score with one threshold.
*Rejected because:* it conflates "what is this about" with "how sure am I",
and the measurement in #6 showed the confidence half carries little signal.

**6. Escalation gates on measured retrieval evidence, not the model's
self-reported confidence.**
*Reason:* The first version gated on the confidence number the LLM emits in
its own JSON, threshold 0.6. Measurement across the test set: the model only
ever emitted 0.80, 0.90 and 0.95 -- the trigger never fired once, and the
number is a token the model chose, not a calibrated probability. Replaced with
cosine similarity of the best-matching historical exchange, thresholded at
0.80 -- the 10th percentile of top-1 similarity measured on the REFERENCE
corpus (1500 probe messages, own row excluded). Reading the threshold off the
golden set, as a first pass did, would tune it on the same examples used to
report the final number.
*Alternative:* keep self-reported confidence as the primary signal.
*Rejected because:* `scripts/calibration.py` tests both signals against
correctness rather than assuming either works. Self-report is retained only as
a weak last check, and the report states what it is.

**7. Retrieval index excludes the golden set AND every message from the same
conversation thread.**
*Reason:* An earlier build indexed the entire corpus including the evaluation
examples. Verified consequence: querying with a test message returned that
same message at similarity 1.0000, so the copy-nearest-reply baseline emitted
the literal ground-truth reply and the agent was handed the real answer as
grounding evidence.
*Alternative:* exclude only exact-match messages.
*Rejected because:* two turns of the same conversation describe the same
incident, so a sibling turn leaks the same answer through a different row.
Excluding by thread costs 88 rows out of 21,681 and closes the hole.

**8. Golden set is stratified into four tagged populations, never blended into
one headline number.**
*Reason:* `natural` (random, the only slice that predicts production),
`rare_boost` (even across clusters, so rare intents have a measurable
per-class score), `escalation_sensitive` (billing/account/fraud language),
`adversarial` (sarcasm, shouting, multi-question).
*Alternative:* one uniform random sample of 200.
*Rejected because:* the first 60-example round drew exactly ONE billing
example. The highest-cost decision in the system was being measured on a
sample of one.

**9. Evaluation restricted to conversation openers.**
*Reason:* 16 of the first 60 golden examples were mid-thread customer replies
("It is usually during ads, yes.") whose meaning lives in a previous turn the
agent never sees. They are unclassifiable in isolation and were depressing
every intent metric for a reason unrelated to classifier quality.
*Alternative:* keep them and pass conversation history into the agent.
*Rejected because:* it changes the task from the one the assignment
describes (classify an incoming message) and adds context-management
complexity that the evaluation would then be measuring instead. Documented as
a real limitation and a one-more-week item.

**10. The simple baseline is given labelled training data the agent never gets.**
*Reason:* TF-IDF + logistic regression needs labels to exist. It trains on the
30% dev split; the LLM agent trains on nothing.
*Alternative:* an untrained keyword-matching baseline, or training the
baseline on fewer examples.
*Rejected because:* it biases the comparison against our own system, which is
the honest direction. A deliberately weak baseline cannot answer whether the
LLM earns its complexity.

**11. Judge model is a different family from the generator.**
*Reason:* Replies from qwen2.5, scored by llama3.1. A judge sharing lineage
with the generator is prone to preferring its own stylistic habits.
*Alternative:* use the same model for both, or a single stronger hosted model
for both roles.
*Rejected because:* same-model judging is the most common way an LLM-judge
result becomes unfalsifiable, and running locally makes the separation free.

**12. Judge scores five defined dimensions, not one overall number.**
*Reason:* grounding, correctness, relevance, safety, tone -- each with a
definition specific enough that two readers would score alike. Grounding and
safety exist because they are the failure modes a fluent-sounding wrong reply
passes on any single "quality" score.
*Alternative:* one 1-5 quality rating.
*Rejected because:* a confident, well-written reply that invents a refund
policy scores well on "quality" and catastrophically on "safety". Collapsing
them hides exactly the error that matters most.

**13. Judge trustworthiness is measured against blind human ratings, not
asserted.**
*Reason:* The human rater sees the customer message and the reply with the
producing system and the judge's score both hidden, so the rating is
independent.
*Alternative:* report judge scores as the reply-quality result.
*Rejected because:* an LLM score is not evidence until something anchors it.
Whatever the agreement number turns out to be, it is reported -- including if
it is poor.

**14. Golden labels are human-made, with no model suggestion ever displayed.**
*Reason:* The labelling tool shows the message and the taxonomy, nothing else.
*Alternative:* pre-fill a model prediction for the annotator to accept or
correct -- far faster.
*Rejected because:* annotators agree with a shown suggestion far more often
than they would unprompted, which would make the golden set a measurement of
the model rather than an independent check on it.

**15. Retrieval + prompting, no fine-tuning.**
*Reason:* 21.7k exchanges from one brand, with the grounding requirement being
"reply the way this brand historically did". Retrieval satisfies that
directly, is inspectable (every reply keeps the exchanges it was grounded in),
and updates instantly when the corpus changes.
*Alternative:* fine-tune a small model on the Hulu exchanges.
*Rejected because:* it would bake 2017 support policy into weights, remove
the ability to show *why* a reply was produced, and cost far more to iterate
on -- with no evidence it would beat retrieval at this corpus size.

**16. Everything runs locally with every call cached to disk.**
*Reason:* Zero marginal cost per experiment, deterministic (`temperature=0`,
fixed seed), and `make reproduce` replays the full result set from cache with
no model calls -- which is what makes the 15-minute reproduction claim honest.
*Alternative:* a hosted API with a stronger model.
*Rejected because:* it would make reproduction depend on a reviewer's API key
and budget, and would tempt using one vendor for both generation and judging.
