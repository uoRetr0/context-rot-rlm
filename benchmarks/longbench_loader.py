"""Load QASPER and NarrativeQA for long-document QA evaluation."""

from __future__ import annotations

import io
import json
import logging
import tarfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

from src.config import settings

logger = logging.getLogger(__name__)

_QASPER_TEST_URL = (
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/"
    "qasper-test-and-evaluator-v0.3.tgz"
)
_QASPER_CACHE = Path("data/qasper_test.json")


@dataclass
class LongBenchSample:
    document: str
    question: str
    answer: str
    dataset_name: str
    sample_id: str


def _download_qasper_test() -> list[dict]:
    """Download and cache QASPER test set from AI2 S3."""
    if _QASPER_CACHE.exists():
        logger.info("Loading cached QASPER from %s", _QASPER_CACHE)
        with open(_QASPER_CACHE) as f:
            return json.load(f)

    logger.info("Downloading QASPER test set from %s", _QASPER_TEST_URL)
    resp = urlopen(_QASPER_TEST_URL)  # noqa: S310
    buf = io.BytesIO(resp.read())

    papers = None
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".json") and "test" in member.name.lower():
                f = tar.extractfile(member)
                if f is not None:
                    papers = json.load(f)
                    break

    if papers is None:
        raise RuntimeError("Could not find test JSON inside QASPER archive")

    # papers is a dict keyed by paper_id
    if isinstance(papers, dict):
        paper_list = [{"id": k, **v} for k, v in papers.items()]
    else:
        paper_list = papers

    _QASPER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(_QASPER_CACHE, "w") as f:
        json.dump(paper_list, f)
    logger.info("Cached %d QASPER papers to %s", len(paper_list), _QASPER_CACHE)
    return paper_list


def load_qasper(max_samples: int | None = None) -> list[LongBenchSample]:
    """Load QASPER dataset (scientific paper QA)."""
    max_samples = max_samples or settings.benchmark_cfg["longbench"]["max_samples"]

    papers = _download_qasper_test()

    samples = []
    for idx, item in enumerate(papers):
        if len(samples) >= max_samples:
            break

        # Build document from paper sections
        full_text_parts = [item.get("title", ""), item.get("abstract", "")]
        for section in item.get("full_text", []):
            if isinstance(section, dict):
                for para in section.get("paragraphs", []):
                    full_text_parts.append(str(para))
            elif isinstance(section, list):
                full_text_parts.extend(str(p) for p in section)
            else:
                full_text_parts.append(str(section))
        full_text = "\n\n".join(str(p) for p in full_text_parts if p)

        # Extract QA pairs
        for qa in item.get("qas", []):
            question = qa.get("question", "")

            for ans_obj in qa.get("answers", []):
                if not isinstance(ans_obj, dict):
                    continue
                answer_inner = ans_obj.get("answer", ans_obj)
                if not isinstance(answer_inner, dict):
                    continue

                unanswerable = answer_inner.get("unanswerable", False)
                if unanswerable:
                    continue

                free_form = answer_inner.get("free_form_answer", "")
                extractive = answer_inner.get("extractive_spans", [])
                yes_no = answer_inner.get("yes_no")

                if free_form:
                    answer = free_form
                elif extractive:
                    answer = " ".join(extractive)
                elif yes_no is not None:
                    answer = "yes" if yes_no else "no"
                else:
                    continue

                if question and answer and full_text:
                    samples.append(LongBenchSample(
                        document=full_text,
                        question=question,
                        answer=answer,
                        dataset_name="qasper",
                        sample_id=f"qasper_{idx}_{len(samples)}",
                    ))
                    break  # One answer per question

            if len(samples) >= max_samples:
                break

    logger.info("Loaded %d QASPER samples", len(samples))
    return samples


def load_narrativeqa(max_samples: int | None = None) -> list[LongBenchSample]:
    """Load NarrativeQA dataset."""
    from datasets import load_dataset

    max_samples = max_samples or settings.benchmark_cfg["longbench"]["max_samples"]

    ds = load_dataset("deepmind/narrativeqa", split="test")

    samples = []
    for idx, item in enumerate(ds):
        if len(samples) >= max_samples:
            break

        document = item.get("document", {})
        doc_text = document.get("summary", {}).get("text", "")

        question = item.get("question", {}).get("text", "")
        answers = item.get("answers", [])

        if answers:
            answer = answers[0].get("text", "") if isinstance(answers[0], dict) else str(answers[0])
        else:
            continue

        if doc_text and question and answer:
            samples.append(LongBenchSample(
                document=doc_text,
                question=question,
                answer=answer,
                dataset_name="narrativeqa",
                sample_id=f"narrativeqa_{idx}",
            ))

    logger.info("Loaded %d NarrativeQA samples", len(samples))
    return samples


def load_longbench(
    dataset_names: list[str] | None = None,
    max_samples: int | None = None,
) -> list[LongBenchSample]:
    """Load all configured LongBench datasets."""
    names = dataset_names or settings.benchmark_cfg["longbench"]["datasets"]
    all_samples = []

    loaders = {
        "qasper": load_qasper,
        "narrativeqa": load_narrativeqa,
    }

    for name in names:
        loader = loaders.get(name)
        if loader is None:
            logger.warning("Unknown dataset: %s", name)
            continue
        samples = loader(max_samples=max_samples)
        all_samples.extend(samples)

    return all_samples
