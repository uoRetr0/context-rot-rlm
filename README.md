# context-rot-rlm

A reproduction and evaluation of Recursive Language Models (Zhang, Kraska, and Khattab, [arXiv:2512.24601](https://arxiv.org/abs/2512.24601)). The question: when the answer is buried deep in a long document and models start missing it (context rot), does a recursive controller that searches, reads small spans, and asks itself sub-questions beat the standard alternatives?

Short answer: sometimes. The full writeup is in [`report/context_rot_report.pdf`](report/context_rot_report.pdf).

## Methods compared

All four methods run under the same harness, budget tracking, and scoring, using the Gemini API.

| Method | How it answers |
|---|---|
| Full Context | Stuffs up to 900k tokens of source text into one prompt |
| RAG | Retrieves top-k chunks once (BM25 + embeddings, reciprocal-rank fusion), answers once |
| Map-Reduce | Summarizes retrieved chunks, then answers from the summaries |
| RLM | A bounded REPL-style controller with tools: `search`, `read_chunks`, `read_span`, `chunk_span`, recursive `llm_query`, and `finish`. Max depth 4, max 8 steps |

Models: `gemini-2.0-flash` (fast), `gemini-2.5-pro` (Pro follow-up runs), `gemini-embedding-001` (retrieval).

## Benchmarks

- Needle-in-Haystack: haystacks up to 500k tokens, 7 needle positions, plus stress variants with repeated distractors and confusion blocks
- Synthetic multi-hop QA: 2 and 3 hops, documents from 10k to 500k tokens
- LongBench: the `qasper` and `narrativeqa` subsets
- MuSiQue: 2, 3, and 4-hop questions

## Results

2,003 of 2,020 evaluations were retained, at $10.03 of tracked API cost. No method wins everywhere.

| Benchmark | Full Context | RAG | Map-Reduce | RLM |
|---|---|---|---|---|
| Needle-in-Haystack (mean F1) | 0.900 | 0.900 | 0.896 | **0.939** |
| Synthetic multi-hop | 0.450 | **0.795** | 0.317 | 0.666 |
| LongBench | **0.543** | 0.513 | 0.526 | 0.484 |
| MuSiQue | - | **0.445** | - | 0.352 |

The interesting slice is 3-hop questions over 500k-token documents, where context rot actually bites. There RLM leads at 0.960 F1 while Full Context drops to 0.800, and at 2-hop/500k Full Context collapses to 0.067 while RAG holds 0.867 and RLM 0.733.

So this reproduction supports the RLM paper's intuition more than its strongest claims. RLM's wins come with higher latency and more API calls, and one-shot RAG remains the better default on multi-hop retrieval tasks. The report includes a failure analysis of the zero-scoring Map-Reduce slices, which traced to a benchmark formulation issue rather than a pipeline bug.

## Running it

Requires Python 3.11+ and a Gemini API key.

```bash
pip install -r requirements.txt
echo "GOOGLE_API_KEY=your-key" > .env

# Full experiment suite (all benchmarks, then plots). Costs real money; see budget caps in config.yaml.
python -m experiments.runner

# Aggregate results into summary tables
python -m analysis.analyze

# Regenerate plots
python -m analysis.plots

# Test suite: 53 tests covering retrieval, benchmarks, the RLM controller, and the runner
pytest
```

All knobs (models, chunking, retrieval depth, RLM step limits, benchmark sizes, dollar budget) live in `config.yaml`. Runs save incrementally per method, log a trace for every sample, retry transient API failures, and stop at the configured spending cap.

## Repo layout

```
src/
  controllers/    Full Context, RAG, Map-Reduce, and RLM controllers
  environment/    document store, BM25 index, vector index, hybrid retriever
  tools/          search / read / reason tools exposed to the RLM controller
  trace/          per-sample trace logging and a trace viewer
  gemini_client.py, cost_tracker.py, config.py
benchmarks/       needle-in-haystack, synthetic multi-hop, LongBench and MuSiQue loaders, F1 metrics
experiments/      runner.py and experiment configs
analysis/         result aggregation and plotting
notebooks/        final report notebook
report/           LaTeX source and compiled PDF
tests/            53 pytest tests
```
