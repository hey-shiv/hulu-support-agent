PY = .venv/bin/python

.PHONY: help reproduce all test data brand cluster golden label index silver eval metrics calibration judge-validity rate agreement clean-cache

help:
	@echo "make reproduce   - replay headline results from cache (no model calls, ~1 min)"
	@echo "make all         - full pipeline from raw data (slow, calls models)"
	@echo "make test        - run the test suite"
	@echo ""
	@echo "individual stages: data brand cluster golden label index eval metrics calibration rate agreement"

# Replays every cached model call. This is the entry point a reviewer should use.
reproduce: metrics calibration judge-validity
	@echo ""
	@echo "Reply-quality numbers above come from an LLM judge."
	@echo "Run 'make agreement' for how far that judge tracks a human."

all: data brand cluster golden index silver eval metrics calibration judge-validity

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
