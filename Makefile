PY = .venv/bin/python

.PHONY: help reproduce all test data brand cluster golden label index silver eval metrics calibration judge-validity rate agreement clean-cache

help:
	@echo "make reproduce   - replay headline results from cache (no model calls, ~1 min)"
	@echo "make all         - full pipeline from raw data (slow, calls models)"
	@echo "make test        - run the test suite"
	@echo ""
	@echo "stages: data brand cluster index silver eval metrics calibration judge-validity"
	@echo ""
	@echo "human-in-the-loop (not run by `all`):"
	@echo "  golden  - draw more evaluation examples (appends unlabelled rows)"
	@echo "  label   - hand-label them"
	@echo "  rate    - blind-rate replies for judge agreement (NOT YET DONE)"

# Replays every cached model call. This is the entry point a reviewer should use.
reproduce: metrics calibration judge-validity
	@echo ""
	@echo "Reply-quality numbers above come from an LLM judge."
	@echo "Run 'make agreement' for how far that judge tracks a human."

# NOTE: `golden` is deliberately NOT here. It appends new unlabelled rows to
# the hand-labelled set, which would corrupt existing human labels. It is a
# one-time step, run manually and followed by `make label`.
all: data brand cluster index silver eval metrics calibration judge-validity

test:
	$(PY) -m pytest tests/ -q

data:
	$(PY) -c "import sys; sys.path.insert(0,'.'); \
	from src.data_prep import load_raw, build_pairs; \
	p = build_pairs(load_raw()); p.to_parquet('data/processed/pairs.parquet', index=False); \
	print(f'{len(p):,} exchanges')"

brand:
	$(PY) scripts/select_brand.py

cluster:
	$(PY) scripts/taxonomy.py

golden:
	$(PY) scripts/sample_golden.py

# Human step. Cannot be automated without invalidating the evaluation.
label:
	$(PY) -m src.label_tui

index:
	$(PY) scripts/build_index.py

eval:
	$(PY) scripts/run_eval.py

metrics:
	$(PY) scripts/metrics.py

calibration:
	$(PY) scripts/calibration.py

judge-validity:
	$(PY) scripts/judge_validity.py

silver:
	$(PY) scripts/make_silver.py

# Human step. Blind rating of replies, to measure judge trustworthiness.
rate:
	$(PY) -m src.rate_tui

agreement:
	$(PY) scripts/judge_agreement.py

clean-cache:
	rm -rf cache/llm cache/embeddings
