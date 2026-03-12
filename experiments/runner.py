"""Experiment runner: load → run → score → save."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from benchmarks.longbench_loader import LongBenchSample, load_longbench
from benchmarks.metrics import compute_all_metrics
from benchmarks.musique_loader import MuSiQueSample, load_musique
from benchmarks.multihop_synthetic import MultihopSample, generate_benchmark as gen_multihop
from benchmarks.needle_haystack import NeedleSample, generate_benchmark as gen_needle
from src.config import PROJECT_ROOT, settings
from src.controllers.base import BaseController, ControllerResult
from src.controllers.fullcontext import FullContextController
from src.controllers.mapreduce import MapReduceController
from src.controllers.rag_baseline import RAGController
from src.controllers.rlm_controller import RLMController
from src.cost_tracker import BudgetExceededError, UsageSnapshot, tracker
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
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0
    status: str = "ok"
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

    retriever = None
    cache_key = ""
    if getattr(controller, "requires_retriever", True):
        doc_hash = hashlib.md5(document.encode("utf-8")).hexdigest()[:12]
        cache_key = f"{sample_id}_{settings.model_embedding}_{store.num_chunks}_{doc_hash}"
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


def _usage_payload(start: UsageSnapshot) -> dict[str, int | float]:
    delta = tracker.delta(start)
    return {
        "input_tokens": delta.input_tokens,
        "output_tokens": delta.output_tokens,
        "llm_calls": delta.calls,
        "cost_usd": delta.cost_usd,
    }


def _classify_error(exc: Exception) -> str:
    err = str(exc)
    transient_markers = (
        "429",
        "RESOURCE_EXHAUSTED",
        "503",
        "UNAVAILABLE",
        "500 INTERNAL",
        "'status': 'INTERNAL'",
        '"status": "INTERNAL"',
    )
    return "transient_error" if any(marker in err for marker in transient_markers) else "error"


def _load_partial(filename: str) -> tuple[list[SampleResult], set[str]]:
    """Load existing partial results and return (results, completed_methods)."""
    path = RESULTS_DIR / f"{filename}.json"
    if not path.exists():
        return [], set()

    with open(path) as f:
        data = json.load(f)

    results = []
    method_sample_counts: dict[str, int] = {}
    for r in data:
        results.append(SampleResult(
            sample_id=r["sample_id"],
            method=r["method"],
            question=r["question"],
            predicted=r["predicted"],
            reference=r["reference"],
            metrics=r["metrics"],
            confidence=r["confidence"],
            duration_s=r["duration_s"],
            input_tokens=r.get("input_tokens", 0),
            output_tokens=r.get("output_tokens", 0),
            llm_calls=r.get("llm_calls", 0),
            cost_usd=r.get("cost_usd", 0.0),
            status=r.get("status", "ok"),
            metadata=r.get("metadata", {}),
        ))
        method_sample_counts[r["method"]] = method_sample_counts.get(r["method"], 0) + 1

    completed = set(method_sample_counts.keys())
    logger.info(
        "Loaded %d partial results from %s (methods: %s)",
        len(results), filename, ", ".join(f"{m}={c}" for m, c in sorted(method_sample_counts.items())),
    )
    return results, completed


def run_needle_haystack(
    methods: list[str] | None = None,
    model: str | None = None,
    max_samples: int | None = None,
    save_as: str = "needle_haystack",
) -> list[SampleResult]:
    """Run needle-in-haystack benchmark."""
    methods = methods or settings.methods
    samples = gen_needle()
    if max_samples:
        samples = samples[:max_samples]

    results, done_methods = _load_partial(save_as)
    remaining = [m for m in methods if m not in done_methods]
    if not remaining:
        logger.info("needle: all methods already complete, skipping")
        return results

    logger.info("Running needle-haystack: %d samples x %d methods (skipping %s)",
                len(samples), len(remaining), done_methods or "none")

    for method in remaining:
        controller = get_controller(method, model)
        trace_dir = RESULTS_DIR / "traces" / "needle" / method
        trace_dir.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(tqdm(samples, desc=f"needle/{method}")):
            sid = f"needle_{sample.haystack_length}_{sample.needle_position:.2f}_{i}"
            usage_start = tracker.snapshot()
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
                    **_usage_payload(usage_start),
                    metadata={
                        "haystack_length": sample.haystack_length,
                        "needle_position": sample.needle_position,
                    },
                ))
            except BudgetExceededError:
                logger.error("Budget exceeded, stopping.")
                save_results(results, save_as)
                return results
            except Exception as e:
                error_type = _classify_error(e)
                logger.error("Error on %s: %s", sid, e)
                results.append(SampleResult(
                    sample_id=sid, method=method,
                    question=sample.question,
                    predicted="ERROR", reference=sample.answer,
                    metrics={"exact_match": 0, "f1": 0, "rouge_l": 0},
                    confidence=0, duration_s=0,
                    status=error_type,
                    **_usage_payload(usage_start),
                    metadata={
                        "error": str(e),
                        "error_type": error_type,
                        "haystack_length": sample.haystack_length,
                        "needle_position": sample.needle_position,
                    },
                ))

        # Incremental save after each method
        save_results(results, save_as)

    return results


def run_multihop(
    methods: list[str] | None = None,
    model: str | None = None,
    max_samples: int | None = None,
    save_as: str = "multihop",
) -> list[SampleResult]:
    """Run multi-hop benchmark."""
    methods = methods or settings.methods
    samples = gen_multihop()
    if max_samples:
        samples = samples[:max_samples]

    results, done_methods = _load_partial(save_as)
    remaining = [m for m in methods if m not in done_methods]
    if not remaining:
        logger.info("multihop: all methods already complete, skipping")
        return results

    logger.info("Running multihop: %d samples x %d methods (skipping %s)",
                len(samples), len(remaining), done_methods or "none")

    for method in remaining:
        controller = get_controller(method, model)
        trace_dir = RESULTS_DIR / "traces" / "multihop" / method
        trace_dir.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(tqdm(samples, desc=f"multihop/{method}")):
            sid = f"multihop_{sample.hops}hop_{sample.doc_length}w_{i}"
            usage_start = tracker.snapshot()
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
                    **_usage_payload(usage_start),
                    metadata={"hops": sample.hops, "doc_length": sample.doc_length},
                ))
            except BudgetExceededError:
                logger.error("Budget exceeded, stopping.")
                save_results(results, save_as)
                return results
            except Exception as e:
                error_type = _classify_error(e)
                logger.error("Error on %s: %s", sid, e)
                results.append(SampleResult(
                    sample_id=sid, method=method,
                    question=sample.question,
                    predicted="ERROR", reference=sample.answer,
                    metrics={"exact_match": 0, "f1": 0, "rouge_l": 0},
                    confidence=0, duration_s=0,
                    status=error_type,
                    **_usage_payload(usage_start),
                    metadata={"error": str(e), "error_type": error_type, "hops": sample.hops, "doc_length": sample.doc_length},
                ))

        # Incremental save after each method
        save_results(results, save_as)

    return results


def run_longbench(
    methods: list[str] | None = None,
    model: str | None = None,
    max_samples: int | None = None,
    save_as: str = "longbench",
) -> list[SampleResult]:
    """Run LongBench (QASPER + NarrativeQA) benchmark."""
    methods = methods or settings.methods
    samples = load_longbench(max_samples=max_samples)

    results, done_methods = _load_partial(save_as)
    remaining = [m for m in methods if m not in done_methods]
    if not remaining:
        logger.info("longbench: all methods already complete, skipping")
        return results

    logger.info("Running longbench: %d samples x %d methods (skipping %s)",
                len(samples), len(remaining), done_methods or "none")

    for method in remaining:
        controller = get_controller(method, model)
        trace_dir = RESULTS_DIR / "traces" / "longbench" / method
        trace_dir.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(tqdm(samples, desc=f"longbench/{method}")):
            sid = sample.sample_id
            usage_start = tracker.snapshot()
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
                    **_usage_payload(usage_start),
                    metadata={"dataset": sample.dataset_name},
                ))
            except BudgetExceededError:
                logger.error("Budget exceeded, stopping.")
                save_results(results, save_as)
                return results
            except Exception as e:
                error_type = _classify_error(e)
                logger.error("Error on %s: %s", sid, e)
                results.append(SampleResult(
                    sample_id=sid, method=method,
                    question=sample.question,
                    predicted="ERROR", reference=sample.answer,
                    metrics={"exact_match": 0, "f1": 0, "rouge_l": 0},
                    confidence=0, duration_s=0,
                    status=error_type,
                    **_usage_payload(usage_start),
                    metadata={
                        "error": str(e),
                        "error_type": error_type,
                        "dataset": sample.dataset_name,
                    },
                ))

        # Incremental save after each method
        save_results(results, save_as)

    return results


def _best_metrics(prediction: str, answer: str, aliases: list[str]) -> dict[str, float]:
    """Compute metrics against answer and all aliases, return the best F1."""
    best = compute_all_metrics(prediction, answer)
    for alias in aliases:
        alt = compute_all_metrics(prediction, alias)
        if alt["f1"] > best["f1"]:
            best = alt
    return best


def run_musique(
    methods: list[str] | None = None,
    model: str | None = None,
    max_samples: int | None = None,
    save_as: str = "musique",
) -> list[SampleResult]:
    """Run MuSiQue multi-hop benchmark (RAG vs RLM stress test)."""
    cfg = settings.benchmark_cfg.get("musique", {})
    methods = methods or cfg.get("methods", ["rag", "rlm"])
    samples = load_musique(max_samples=max_samples)

    results, done_methods = _load_partial(save_as)
    remaining = [m for m in methods if m not in done_methods]
    if not remaining:
        logger.info("musique: all methods already complete, skipping")
        return results

    logger.info("Running MuSiQue: %d samples x %d methods (skipping %s)",
                len(samples), len(remaining), done_methods or "none")

    for method in remaining:
        controller = get_controller(method, model)
        trace_dir = RESULTS_DIR / "traces" / "musique" / method
        trace_dir.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(tqdm(samples, desc=f"musique/{method}")):
            sid = sample.sample_id
            usage_start = tracker.snapshot()
            try:
                result, elapsed = _run_single(
                    controller, sample.document, sample.question, sid, trace_dir
                )
                metrics = _best_metrics(
                    result.answer, sample.answer, sample.answer_aliases
                )
                results.append(SampleResult(
                    sample_id=sid, method=method,
                    question=sample.question,
                    predicted=result.answer, reference=sample.answer,
                    metrics=metrics, confidence=result.confidence,
                    duration_s=elapsed,
                    **_usage_payload(usage_start),
                    metadata={
                        "hops": sample.hops,
                        "doc_length": sample.doc_length,
                        "bridge_entities": sample.bridge_entities,
                        "sub_questions": sample.sub_questions,
                    },
                ))
            except BudgetExceededError:
                logger.error("Budget exceeded, stopping.")
                save_results(results, save_as)
                return results
            except Exception as e:
                error_type = _classify_error(e)
                logger.error("Error on %s: %s", sid, e)
                results.append(SampleResult(
                    sample_id=sid, method=method,
                    question=sample.question,
                    predicted="ERROR", reference=sample.answer,
                    metrics={"exact_match": 0, "f1": 0, "rouge_l": 0},
                    confidence=0, duration_s=0,
                    status=error_type,
                    **_usage_payload(usage_start),
                    metadata={
                        "error": str(e),
                        "error_type": error_type,
                        "hops": sample.hops,
                        "doc_length": sample.doc_length,
                    },
                ))

        # Incremental save after each method
        save_results(results, save_as)

    return results


def run_pro_musique(
    phase: str = "A",
    max_samples: int | None = None,
) -> list[SampleResult]:
    """Run MuSiQue with Pro model in phases to stay within quota.

    Phase A: RAG on all hops/lengths + RLM on 2-hop only (~1260 calls)
    Phase B: RLM on 3-hop only (~1500 calls)
    Skip 4-hop RLM (Flash showed degradation, quota too tight).
    """
    cfg = settings.benchmark_cfg.get("musique", {})
    n = max_samples or cfg.get("num_samples", 10)
    samples = load_musique(max_samples=n)
    model = settings.model_pro

    results: list[SampleResult] = []

    if phase == "A":
        # RAG on all samples
        controller = get_controller("rag", model)
        trace_dir = RESULTS_DIR / "traces" / "musique_pro" / "rag"
        trace_dir.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(tqdm(samples, desc="musique_pro/rag")):
            sid = sample.sample_id
            usage_start = tracker.snapshot()
            try:
                result, elapsed = _run_single(
                    controller, sample.document, sample.question, sid, trace_dir
                )
                metrics = _best_metrics(result.answer, sample.answer, sample.answer_aliases)
                results.append(SampleResult(
                    sample_id=sid, method="rag",
                    question=sample.question,
                    predicted=result.answer, reference=sample.answer,
                    metrics=metrics, confidence=result.confidence,
                    duration_s=elapsed,
                    **_usage_payload(usage_start),
                    metadata={
                        "hops": sample.hops,
                        "doc_length": sample.doc_length,
                        "bridge_entities": sample.bridge_entities,
                        "sub_questions": sample.sub_questions,
                        "model": model,
                    },
                ))
            except BudgetExceededError:
                logger.error("Budget exceeded, stopping.")
                save_results(results, "musique_pro_phaseA_rag")
                return results
            except Exception as e:
                error_type = _classify_error(e)
                logger.error("Error on %s: %s", sid, e)
                results.append(SampleResult(
                    sample_id=sid, method="rag",
                    question=sample.question,
                    predicted="ERROR", reference=sample.answer,
                    metrics={"exact_match": 0, "f1": 0, "rouge_l": 0},
                    confidence=0, duration_s=0,
                    status=error_type,
                    **_usage_payload(usage_start),
                    metadata={
                        "error": str(e), "error_type": error_type,
                        "hops": sample.hops, "doc_length": sample.doc_length,
                        "model": model,
                    },
                ))

        save_results(results, "musique_pro_phaseA_rag")

        # RLM on 2-hop only
        rlm_samples = [s for s in samples if s.hops == 2]
        controller = get_controller("rlm", model)
        trace_dir = RESULTS_DIR / "traces" / "musique_pro" / "rlm"
        trace_dir.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(tqdm(rlm_samples, desc="musique_pro/rlm_2hop")):
            sid = sample.sample_id
            usage_start = tracker.snapshot()
            try:
                result, elapsed = _run_single(
                    controller, sample.document, sample.question, sid, trace_dir
                )
                metrics = _best_metrics(result.answer, sample.answer, sample.answer_aliases)
                results.append(SampleResult(
                    sample_id=sid, method="rlm",
                    question=sample.question,
                    predicted=result.answer, reference=sample.answer,
                    metrics=metrics, confidence=result.confidence,
                    duration_s=elapsed,
                    **_usage_payload(usage_start),
                    metadata={
                        "hops": sample.hops,
                        "doc_length": sample.doc_length,
                        "bridge_entities": sample.bridge_entities,
                        "sub_questions": sample.sub_questions,
                        "model": model,
                    },
                ))
            except BudgetExceededError:
                logger.error("Budget exceeded, stopping.")
                save_results(results, "musique_pro_phaseA")
                return results
            except Exception as e:
                error_type = _classify_error(e)
                logger.error("Error on %s: %s", sid, e)
                results.append(SampleResult(
                    sample_id=sid, method="rlm",
                    question=sample.question,
                    predicted="ERROR", reference=sample.answer,
                    metrics={"exact_match": 0, "f1": 0, "rouge_l": 0},
                    confidence=0, duration_s=0,
                    status=error_type,
                    **_usage_payload(usage_start),
                    metadata={
                        "error": str(e), "error_type": error_type,
                        "hops": sample.hops, "doc_length": sample.doc_length,
                        "model": model,
                    },
                ))

        save_results(results, "musique_pro_phaseA")

    elif phase == "B":
        # RLM on 3-hop only
        rlm_samples = [s for s in samples if s.hops == 3]
        controller = get_controller("rlm", model)
        trace_dir = RESULTS_DIR / "traces" / "musique_pro" / "rlm"
        trace_dir.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(tqdm(rlm_samples, desc="musique_pro/rlm_3hop")):
            sid = sample.sample_id
            usage_start = tracker.snapshot()
            try:
                result, elapsed = _run_single(
                    controller, sample.document, sample.question, sid, trace_dir
                )
                metrics = _best_metrics(result.answer, sample.answer, sample.answer_aliases)
                results.append(SampleResult(
                    sample_id=sid, method="rlm",
                    question=sample.question,
                    predicted=result.answer, reference=sample.answer,
                    metrics=metrics, confidence=result.confidence,
                    duration_s=elapsed,
                    **_usage_payload(usage_start),
                    metadata={
                        "hops": sample.hops,
                        "doc_length": sample.doc_length,
                        "bridge_entities": sample.bridge_entities,
                        "sub_questions": sample.sub_questions,
                        "model": model,
                    },
                ))
            except BudgetExceededError:
                logger.error("Budget exceeded, stopping.")
                save_results(results, "musique_pro_phaseB")
                return results
            except Exception as e:
                error_type = _classify_error(e)
                logger.error("Error on %s: %s", sid, e)
                results.append(SampleResult(
                    sample_id=sid, method="rlm",
                    question=sample.question,
                    predicted="ERROR", reference=sample.answer,
                    metrics={"exact_match": 0, "f1": 0, "rouge_l": 0},
                    confidence=0, duration_s=0,
                    status=error_type,
                    **_usage_payload(usage_start),
                    metadata={
                        "error": str(e), "error_type": error_type,
                        "hops": sample.hops, "doc_length": sample.doc_length,
                        "model": model,
                    },
                ))

        save_results(results, "musique_pro_phaseB")

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
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "llm_calls": r.llm_calls,
            "cost_usd": r.cost_usd,
            "status": r.status,
            "metadata": r.metadata,
        })

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Saved %d results to %s", len(results), out_path)
    return out_path


def run_all_experiments() -> None:
    """Run all benchmarks and save results.

    Resumable: each benchmark saves incrementally after each method.
    Re-running skips already-completed methods automatically.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # 1. Needle-in-haystack
    logger.info("=== Needle-in-Haystack ===")
    run_needle_haystack()

    # 2. Multi-hop
    logger.info("=== Multi-hop ===")
    run_multihop()

    # 3. LongBench
    logger.info("=== LongBench ===")
    run_longbench()

    # 4. MuSiQue (RAG vs RLM stress test)
    logger.info("=== MuSiQue ===")
    run_musique()

    # 5. RLM with Pro model (subset)
    logger.info("=== RLM Pro ===")
    run_needle_haystack(
        methods=["rlm"], model=settings.model_pro, max_samples=20,
        save_as="needle_rlm_pro",
    )

    # 6. Pro MuSiQue Phase A (RAG all + RLM 2-hop)
    logger.info("=== Pro MuSiQue Phase A ===")
    run_pro_musique(phase="A")
    # Phase B (RLM 3-hop) — run separately on day 2:
    # run_pro_musique(phase="B")

    # Cost summary
    print("\n" + tracker.summary())
    cost_path = RESULTS_DIR / "cost_summary.txt"
    cost_path.write_text(tracker.summary())


if __name__ == "__main__":
    run_all_experiments()
