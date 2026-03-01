"""Experiment runner: load → run → score → save."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from benchmarks.longbench_loader import LongBenchSample, load_longbench
from benchmarks.metrics import compute_all_metrics
from benchmarks.multihop_synthetic import MultihopSample, generate_benchmark as gen_multihop
from benchmarks.needle_haystack import NeedleSample, generate_benchmark as gen_needle
from src.config import PROJECT_ROOT, settings
from src.controllers.base import BaseController, ControllerResult
from src.controllers.fullcontext import FullContextController
from src.controllers.mapreduce import MapReduceController
from src.controllers.rag_baseline import RAGController
from src.controllers.rlm_controller import RLMController
from src.cost_tracker import BudgetExceededError, tracker
from src.environment.document_store import DocumentStore
from src.environment.hybrid_retriever import HybridRetriever
from src.trace.trace_viewer import export_trace

logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / settings.output_dir


def get_controller(method: str, model: str | None = None) -> BaseController:
    """Factory for controllers."""
    controllers = {
        "fullcontext": FullContextController,
        "rag": RAGController,
        "mapreduce": MapReduceController,
        "rlm": RLMController,
    }
    cls = controllers[method]
    return cls(model=model) if model else cls()


@dataclass
class SampleResult:
    sample_id: str
    method: str
    question: str
    predicted: str
    reference: str
    metrics: dict[str, float]
    confidence: float
    duration_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _run_single(
    controller: BaseController,
    document: str,
    question: str,
    sample_id: str,
    trace_dir: Path | None = None,
) -> tuple[ControllerResult, float]:
    """Run a single QA sample through a controller."""
    store = DocumentStore()
    store.ingest(document, doc_id=sample_id)

    # Build retriever once and pass to controller
    cache_key = f"{sample_id}_{controller.method_name}"
    retriever = HybridRetriever(store, cache_key=cache_key)

    start = time.time()
    result = controller.answer(
        question, store, retriever=retriever, cache_key=cache_key
    )
    elapsed = time.time() - start

    # Export trace
    if trace_dir and result.trace:
        export_trace(result.trace, trace_dir / f"{sample_id}.json")

    return result, elapsed


def run_needle_haystack(
    methods: list[str] | None = None,
    model: str | None = None,
    max_samples: int | None = None,
) -> list[SampleResult]:
    """Run needle-in-haystack benchmark."""
    methods = methods or settings.methods
    samples = gen_needle()
    if max_samples:
        samples = samples[:max_samples]

    logger.info("Running needle-haystack: %d samples x %d methods", len(samples), len(methods))
    results = []

    for method in methods:
        controller = get_controller(method, model)
        trace_dir = RESULTS_DIR / "traces" / "needle" / method
        trace_dir.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(tqdm(samples, desc=f"needle/{method}")):
            sid = f"needle_{sample.haystack_length}_{sample.needle_position:.2f}_{i}"
            try:
                result, elapsed = _run_single(
                    controller, sample.document, sample.question, sid, trace_dir
                )
                metrics = compute_all_metrics(result.answer, sample.answer)
                results.append(SampleResult(
                    sample_id=sid, method=method,
                    question=sample.question,
                    predicted=result.answer, reference=sample.answer,
                    metrics=metrics, confidence=result.confidence,
                    duration_s=elapsed,
                    metadata={
                        "haystack_length": sample.haystack_length,
                        "needle_position": sample.needle_position,
                    },
                ))
            except BudgetExceededError:
                logger.error("Budget exceeded, stopping.")
                return results
            except Exception as e:
                logger.error("Error on %s: %s", sid, e)
                results.append(SampleResult(
                    sample_id=sid, method=method,
                    question=sample.question,
                    predicted="ERROR", reference=sample.answer,
                    metrics={"exact_match": 0, "f1": 0, "rouge_l": 0},
                    confidence=0, duration_s=0,
                    metadata={"error": str(e)},
                ))

    return results


def run_multihop(
    methods: list[str] | None = None,
    model: str | None = None,
    max_samples: int | None = None,
) -> list[SampleResult]:
    """Run multi-hop benchmark."""
    methods = methods or settings.methods
    samples = gen_multihop()
    if max_samples:
        samples = samples[:max_samples]

    logger.info("Running multihop: %d samples x %d methods", len(samples), len(methods))
    results = []

    for method in methods:
        controller = get_controller(method, model)
        trace_dir = RESULTS_DIR / "traces" / "multihop" / method
        trace_dir.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(tqdm(samples, desc=f"multihop/{method}")):
            sid = f"multihop_{sample.hops}hop_{i}"
            try:
                result, elapsed = _run_single(
                    controller, sample.document, sample.question, sid, trace_dir
                )
                metrics = compute_all_metrics(result.answer, sample.answer)
                results.append(SampleResult(
                    sample_id=sid, method=method,
                    question=sample.question,
                    predicted=result.answer, reference=sample.answer,
                    metrics=metrics, confidence=result.confidence,
                    duration_s=elapsed,
                    metadata={"hops": sample.hops},
                ))
            except BudgetExceededError:
                logger.error("Budget exceeded, stopping.")
                return results
            except Exception as e:
                logger.error("Error on %s: %s", sid, e)
                results.append(SampleResult(
                    sample_id=sid, method=method,
                    question=sample.question,
                    predicted="ERROR", reference=sample.answer,
                    metrics={"exact_match": 0, "f1": 0, "rouge_l": 0},
                    confidence=0, duration_s=0,
                    metadata={"error": str(e)},
                ))

    return results


def run_longbench(
    methods: list[str] | None = None,
    model: str | None = None,
    max_samples: int | None = None,
) -> list[SampleResult]:
    """Run LongBench (QASPER + NarrativeQA) benchmark."""
    methods = methods or settings.methods
    samples = load_longbench(max_samples=max_samples)

    logger.info("Running longbench: %d samples x %d methods", len(samples), len(methods))
    results = []

    for method in methods:
        controller = get_controller(method, model)
        trace_dir = RESULTS_DIR / "traces" / "longbench" / method
        trace_dir.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(tqdm(samples, desc=f"longbench/{method}")):
            sid = sample.sample_id
            try:
                result, elapsed = _run_single(
                    controller, sample.document, sample.question, sid, trace_dir
                )
                metrics = compute_all_metrics(result.answer, sample.answer)
                results.append(SampleResult(
                    sample_id=sid, method=method,
                    question=sample.question,
                    predicted=result.answer, reference=sample.answer,
                    metrics=metrics, confidence=result.confidence,
                    duration_s=elapsed,
                    metadata={"dataset": sample.dataset_name},
                ))
            except BudgetExceededError:
                logger.error("Budget exceeded, stopping.")
                return results
            except Exception as e:
                logger.error("Error on %s: %s", sid, e)
                results.append(SampleResult(
                    sample_id=sid, method=method,
                    question=sample.question,
                    predicted="ERROR", reference=sample.answer,
                    metrics={"exact_match": 0, "f1": 0, "rouge_l": 0},
                    confidence=0, duration_s=0,
                    metadata={"error": str(e), "dataset": sample.dataset_name},
                ))

    return results


def save_results(results: list[SampleResult], filename: str) -> Path:
    """Save results to JSON."""
    out_path = RESULTS_DIR / f"{filename}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for r in results:
        data.append({
            "sample_id": r.sample_id,
            "method": r.method,
            "question": r.question,
            "predicted": r.predicted,
            "reference": r.reference,
            "metrics": r.metrics,
            "confidence": r.confidence,
            "duration_s": r.duration_s,
            "metadata": r.metadata,
        })

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Saved %d results to %s", len(results), out_path)
    return out_path


def run_all_experiments() -> None:
    """Run all benchmarks and save results."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # 1. Needle-in-haystack
    logger.info("=== Needle-in-Haystack ===")
    needle_results = run_needle_haystack()
    save_results(needle_results, "needle_haystack")

    # 2. Multi-hop
    logger.info("=== Multi-hop ===")
    multihop_results = run_multihop()
    save_results(multihop_results, "multihop")

    # 3. LongBench
    logger.info("=== LongBench ===")
    longbench_results = run_longbench()
    save_results(longbench_results, "longbench")

    # 4. RLM with Pro model (subset)
    logger.info("=== RLM Pro ===")
    needle_pro = run_needle_haystack(
        methods=["rlm"], model=settings.model_pro, max_samples=20
    )
    save_results(needle_pro, "needle_rlm_pro")

    # Cost summary
    print("\n" + tracker.summary())
    cost_path = RESULTS_DIR / "cost_summary.txt"
    cost_path.write_text(tracker.summary())


if __name__ == "__main__":
    run_all_experiments()
