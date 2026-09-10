# Hulu Support Agent

An AI customer-support agent for `hulu_support`, built from the Kaggle
["Customer Support on Twitter"](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset. It classifies an incoming message, drafts a reply grounded in how
Hulu historically resolved similar problems, and decides whether to auto-send
or hand off to a human — with a stated reason.

The evaluation is the point of this repo, not the agent. Headline numbers,
their confidence intervals, and the reasons not to trust them are all in
[`reports/report.md`](reports/report.md).

---

## Reproduce

Requires Python 3.12 and [Ollama](https://ollama.com).

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt

ollama pull qwen2.5:7b-instruct-q4_K_M    # generator
ollama pull llama3.1:8b-instruct-q4_K_M   # judge (different family, deliberately)
ollama pull nomic-embed-text              # retrieval embeddings

make reproduce
```

`make reproduce` replays every model call from `cache/` and prints the full
metrics table. No GPU, no API key, no network. Takes about 16 seconds.

`make all` rebuilds everything from the raw CSV including model calls (~30 min,
and needs `data/raw/twcs.csv` downloaded from Kaggle).

`make test` runs 33 tests, including regression tests for the data-leakage bug
described below.

---

## Pipeline

```
incoming message
      |
      v
  classify  ------------------> intent (10 classes, or "other")
      |                         + self-reported confidence [weak signal]
      v
  retrieve  ------------------> 3 most similar historical Hulu exchanges
      |                         + similarity of best match [evidence quality]
      v
  draft     ------------------> reply, constrained to the retrieved evidence
      |                         + fabricated-claim check
      v
  escalate? ------------------> auto-handle | escalate + reason code
```

| Stage | Code | Output |
|---|---|---|
| Reconstruct conversations | [`src/data_prep.py`](src/data_prep.py) | 1.26M exchanges from 2.8M tweets |
| Pick the brand | [`scripts/select_brand.py`](scripts/select_brand.py) | deflection-rate table |
| Discover intents | [`scripts/taxonomy.py`](scripts/taxonomy.py) | clusters → [`src/taxonomy.py`](src/taxonomy.py) |
| Sample golden set | [`scripts/sample_golden.py`](scripts/sample_golden.py) | 200 examples, 4 strata |
| **Label (human)** | [`src/label_tui.py`](src/label_tui.py) | all 200 hand-labelled |
| Build retrieval index | [`scripts/build_index.py`](scripts/build_index.py) | reference set, golden excluded |
| Agent + baselines | [`scripts/run_eval.py`](scripts/run_eval.py) | `reports/predictions.csv` |
| Judge validation | [`scripts/judge_validity.py`](scripts/judge_validity.py) | `reports/judge_validity.csv` |
| Metrics | [`scripts/metrics.py`](scripts/metrics.py) | headline table |
| Calibration check | [`scripts/calibration.py`](scripts/calibration.py) | do the uncertainty signals work? |
| Judge agreement status | [`scripts/judge_agreement.py`](scripts/judge_agreement.py) | what agreement evidence exists, and what is missing |

**Labelling is deliberately not automated.** Its entire value is being an
independent check on the system; generating labels with a model would make the
evaluation measure itself. `src/rate_tui.py` exists for the same reason and is
the one deliverable not completed -- see "Known gaps" below.

## Known gaps

Stated here rather than buried, because a reviewer will find them anyway:

1. **The headline result dropped 3x between two golden-set sizes, and that
   is disclosed as a finding, not smoothed over.** All 200 examples are
   hand-labelled (one annotator, no model suggestion ever shown). The
   original 60-example draw gave macro-F1 0.487 / 72.7% accuracy; the full
   200 -- which surfaced the billing/account cases the small draw almost
   entirely missed -- gives macro-F1 0.222 / 28.5% accuracy. See
   `reports/report.md` section 12 for the full account: this is presented as
   the project's most important evidence for the "misleading headline
   number" question, not hidden.
2. **The escalation policy fails when classification fails first.** 58% of
   real billing disputes are never escalated because the classifier assigns
   them a different intent before the sensitive-intent rule can fire. See
   report section 11.
3. **No per-item judge-human agreement.** `scripts/judge_validity.py` provides
   weaker substitute evidence (the judge recovers a quality ordering fixed by
   construction, and catches injected fabrications). Run
   `make agreement` for an explicit statement of what is and is not
   established. `make rate` collects the real thing in ~4 minutes.
3. **Reply generation is not shown to beat retrieval alone** (paired 95% CI
   contains zero). This is reported as a finding, not smoothed over.

---

## Evaluation integrity

The thing most likely to make a reviewer distrust a submission like this is
leakage, so it is stated plainly:

- **The retrieval index excludes every golden example and every message from
  the same conversation thread** (233 rows removed). An earlier build did not,
  and querying with a test message returned that message at similarity
  `1.0000` — handing the agent the ground-truth reply as "evidence". Every
  reply-quality number from that build measured memorisation. Two tests in
  `tests/test_pipeline.py` now fail if this regresses.
- **All 200 human labels are the test set; none were spent on training or
  tuning.** Nothing here was tuned on them: the LLM is prompted not trained,
  the keyword rules were written from the corpus, and the escalation threshold
  was derived from the reference corpus (p10 of top-1 similarity).
- **Baselines train on silver (LLM-labelled) non-golden messages**, never on
  the human labels, and never scored against. The TF-IDF baseline is therefore
  distilling the LLM and cannot meaningfully exceed it -- and even so, the
  agent's lead over it is not statistically established at n=200 (report §10).
- **Openers and mid-thread fragments are reported separately.** 16 of the 200
  examples are mid-thread replies ("it is a roku TV actually") whose meaning
  lives in a turn the agent never sees. Blending them into one number would
  understate the system on its defined task and overstate it on the harder
  one, so both are reported.
- **All 200 golden labels were produced by one human with no model suggestion
  shown**, so the annotator was never anchored to the system being evaluated.

---

## Cost and determinism

Local models only, so marginal cost per run is zero. `temperature=0` and a
fixed seed everywhere; every LLM and embedding call is cached to disk keyed by
a hash of its exact input, so results do not drift between runs and
`make reproduce` needs no model at all.

Approximate cost of one full `make all`: 5 LLM calls per test example
(1 classify, 1 draft, 3 judge -- one per reply candidate) plus one cached
query embedding, so ~300 LLM calls for the 60-example evaluation. On top of
that, a one-off ~21.5k-message embedding pass (~2 minutes batched) and 400
silver-label calls. `scripts/ablate_retrieval.py` adds a further 360 calls and
is not part of `make all`.

---

## Documents

- [`reports/report.md`](reports/report.md) — problem framing, results vs.
  baselines, failure analysis, **what is misleading about the headline
  number**, next steps
- [`reports/decision_log.md`](reports/decision_log.md) — 16 non-obvious
  decisions, each with the alternative considered and why it was rejected
- [`reports/build_log.md`](reports/build_log.md) — bugs hit during development
  and how they were diagnosed

## Credits

- Dataset: [thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (Kaggle)
- Models: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct),
  [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct),
  [nomic-embed-text](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5), all served via Ollama
- Libraries: pandas, scikit-learn (TF-IDF, KMeans, logistic regression, metrics),
  numpy, scipy (Spearman), rich (labelling UI), pytest

All pipeline code in `src/` and `scripts/` was written for this assignment.
No retrieval, agent, or evaluation framework was used — the retrieval is a
cosine-similarity search over a numpy array, deliberately, so every step is
inspectable and explainable.
