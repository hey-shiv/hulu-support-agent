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
metrics table. No GPU, no API key, no network. Roughly one minute.

`make all` rebuilds everything from the raw CSV including model calls (~30 min,
and needs `data/raw/twcs.csv` downloaded from Kaggle).

`make test` runs 25 tests, including regression tests for the data-leakage bug
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
| **Label (human)** | [`src/label_tui.py`](src/label_tui.py) | `data/golden/golden_labelled.csv` |
| Build retrieval index | [`scripts/build_index.py`](scripts/build_index.py) | reference set, golden excluded |
| Agent + baselines | [`scripts/run_eval.py`](scripts/run_eval.py) | `reports/predictions.csv` |
| **Rate replies (human)** | [`src/rate_tui.py`](src/rate_tui.py) | `reports/human_ratings.csv` |
| Metrics | [`scripts/metrics.py`](scripts/metrics.py) | headline table |
| Calibration check | [`scripts/calibration.py`](scripts/calibration.py) | do the uncertainty signals work? |
| Judge agreement | [`scripts/judge_agreement.py`](scripts/judge_agreement.py) | judge vs. human |

The two human steps are deliberately not automated. Their entire value is
being an independent check on the system; generating them with a model would
make the evaluation measure itself.

---

## Evaluation integrity

The thing most likely to make a reviewer distrust a submission like this is
leakage, so it is stated plainly:

- **The retrieval index excludes every golden example and every message from
  the same conversation thread** (207 rows removed). An earlier build did not,
  and querying with a test message returned that message at similarity
  `1.0000` — handing the agent the ground-truth reply as "evidence". Every
  reply-quality number from that build measured memorisation. Two tests in
  `tests/test_pipeline.py` now fail if this regresses.
- **dev / test split is stratified by intent, seeded, and written to disk**
  (`reports/split_dev.csv`, `reports/split_test.csv`). Prompts and thresholds
  were only ever inspected against dev.
- **The simple baseline trains on labelled dev data the LLM agent never sees.**
  This biases the comparison against our own system on purpose.
- **Only conversation openers are evaluated.** Mid-thread replies ("yes,
  during ads") are meaningless in isolation; including them was depressing
  intent metrics for reasons unrelated to the classifier.
- **Golden labels were produced by a human with no model suggestion shown.**

---

## Cost and determinism

Local models only, so marginal cost per run is zero. `temperature=0` and a
fixed seed everywhere; every LLM and embedding call is cached to disk keyed by
a hash of its exact input, so results do not drift between runs and
`make reproduce` needs no model at all.

Approximate cost of one full `make all`: ~7 model calls per test example
(1 classify + 1 draft + 3 judge + retrieval), plus a one-off ~21.5k-message
embedding pass that takes about two minutes batched.

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
