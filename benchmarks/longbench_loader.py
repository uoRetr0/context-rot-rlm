"""Load QASPER and NarrativeQA from HuggingFace datasets."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class LongBenchSample:
    document: str
    question: str
    answer: str
    dataset_name: str
    sample_id: str


def load_qasper(max_samples: int | None = None) -> list[LongBenchSample]:
    """Load QASPER dataset (scientific paper QA)."""
    from datasets import load_dataset

    max_samples = max_samples or settings.benchmark_cfg["longbench"]["max_samples"]

    ds = load_dataset("allenai/qasper", split="test", trust_remote_code=True)

    samples = []
    for idx, item in enumerate(ds):
        if len(samples) >= max_samples:
            break

        # Build document from paper sections
        full_text_parts = [item.get("title", ""), item.get("abstract", "")]
        for section in item.get("full_text", {}).get("paragraphs", []):
            if isinstance(section, list):
                full_text_parts.extend(section)
            else:
                full_text_parts.append(str(section))
        full_text = "\n\n".join(str(p) for p in full_text_parts if p)

        # Extract QA pairs
        for qa in item.get("qas", []):
            question = qa.get("question", "")
            answers_data = qa.get("answers", {})
            answer_texts = answers_data.get("answer", [])

            for ans_obj in answer_texts:
                # Each answer can be free-form, extractive, yes/no, or unanswerable
                if isinstance(ans_obj, dict):
                    free_form = ans_obj.get("free_form_answer", "")
                    extractive = ans_obj.get("extractive_spans", [])
                    yes_no = ans_obj.get("yes_no")
                    unanswerable = ans_obj.get("unanswerable", False)

                    if unanswerable:
                        continue

                    if free_form:
                        answer = free_form
                    elif extractive:
                        answer = " ".join(extractive)
                    elif yes_no is not None:
                        answer = "yes" if yes_no else "no"
                    else:
                        continue
                else:
                    answer = str(ans_obj)

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

    ds = load_dataset("deepmind/narrativeqa", split="test", trust_remote_code=True)

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
