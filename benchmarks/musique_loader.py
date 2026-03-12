"""Load MuSiQue multi-hop QA benchmark and pad documents to target lengths."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)

_CACHE_PATH = Path("data/musique_validation.json")

_FILLER_SENTENCES = [
    "Research in this area has progressed steadily over the past decade.",
    "International cooperation remains essential for addressing these challenges.",
    "New methodologies have emerged to tackle previously intractable problems.",
    "The implications of these findings extend across multiple disciplines.",
    "Funding agencies have increasingly recognized the importance of this work.",
    "Public engagement with these topics continues to grow.",
    "Technical standards in the field have evolved considerably.",
    "Cross-disciplinary collaboration has yielded unexpected insights.",
    "The theoretical framework underlying this work remains robust.",
    "Practical applications are beginning to emerge from basic research.",
    "Several independent teams have replicated these core findings.",
    "The peer review process ensured rigorous validation of results.",
    "Government policy has gradually adapted to incorporate new evidence.",
    "Historical analysis provides important context for contemporary debates.",
    "Statistical techniques have improved the reliability of conclusions.",
    "Environmental factors play a significant role in observed outcomes.",
    "Educational curricula are being updated to reflect recent advances.",
    "The economic implications of these developments are far-reaching.",
    "Ethical considerations remain at the forefront of ongoing discussions.",
    "Longitudinal studies have confirmed earlier preliminary observations.",
    "Technological innovations continue to reshape established practices.",
    "Community-based approaches have shown promising results.",
    "Regulatory frameworks are struggling to keep pace with rapid change.",
    "Comparative studies across regions reveal important differences.",
    "Data availability has dramatically improved in recent years.",
    "Simulation models have become increasingly sophisticated and accurate.",
    "Professional organizations have issued updated guidelines.",
    "Consumer awareness of these issues has risen significantly.",
    "Infrastructure investments are needed to support future growth.",
    "Quality assurance protocols have been strengthened considerably.",
    "Media coverage has brought greater public attention to these topics.",
    "Workforce training programs are adapting to new requirements.",
    "Supply chain disruptions have highlighted systemic vulnerabilities.",
    "Demographic shifts are creating new patterns of demand.",
    "Intellectual property considerations add complexity to collaboration.",
    "Climate variability introduces additional uncertainty into projections.",
    "Digital transformation is accelerating across many sectors.",
    "Stakeholder engagement has become a critical success factor.",
    "Benchmark comparisons help identify best practices across institutions.",
    "Resource allocation decisions require careful cost-benefit analysis.",
]


@dataclass
class MuSiQueSample:
    document: str
    question: str
    answer: str
    answer_aliases: list[str]
    hops: int
    bridge_entities: list[str]
    sub_questions: list[str]
    doc_length: int
    sample_id: str


def _load_musique_raw() -> list[dict]:
    """Load MuSiQue validation set from HuggingFace, cache locally."""
    if _CACHE_PATH.exists():
        logger.info("Loading cached MuSiQue from %s", _CACHE_PATH)
        with open(_CACHE_PATH) as f:
            return json.load(f)

    logger.info("Downloading MuSiQue from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset("dgslibisey/MuSiQue", split="validation")

    raw = []
    for item in ds:
        if not item.get("answerable", False):
            continue
        raw.append({
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "answer_aliases": item.get("answer_aliases", []),
            "paragraphs": item["paragraphs"],
            "question_decomposition": item["question_decomposition"],
        })

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_PATH, "w") as f:
        json.dump(raw, f)
    logger.info("Cached %d answerable MuSiQue samples to %s", len(raw), _CACHE_PATH)
    return raw


def _extract_bridge_entities(decomposition: list[dict]) -> list[str]:
    """Extract bridge entities from question_decomposition.

    Bridge entities are the answers to all sub-questions except the final one,
    since those intermediate answers are needed to formulate later sub-questions
    but are NOT mentioned in the original question.
    """
    entities = []
    for i, step in enumerate(decomposition):
        if i < len(decomposition) - 1:
            ans = step.get("answer", "")
            if ans:
                entities.append(ans)
    return entities


def _extract_sub_questions(decomposition: list[dict]) -> list[str]:
    """Extract sub-question texts from question_decomposition."""
    return [step.get("question", "") for step in decomposition if step.get("question")]


def _get_paragraphs(item: dict) -> list[str]:
    """Extract paragraph texts from a MuSiQue item."""
    paragraphs = item.get("paragraphs", [])
    texts = []
    for p in paragraphs:
        if isinstance(p, dict):
            # MuSiQue paragraphs have 'title' and 'paragraph_text'
            title = p.get("title", "")
            text = p.get("paragraph_text", "")
            if title and text:
                texts.append(f"{title}: {text}")
            elif text:
                texts.append(text)
        elif isinstance(p, str):
            texts.append(p)
    return [t for t in texts if t.strip()]


def _count_hops(decomposition: list[dict]) -> int:
    """Count number of hops from question_decomposition."""
    return len(decomposition)


def _pad_document(paragraphs: list[str], target_words: int, rng: random.Random) -> str:
    """Insert filler text between paragraphs to reach target_words.

    Paragraphs are spread evenly throughout the padded document so that
    RAG retrieval with a fixed top-k can't trivially cover all of them.
    """
    para_words = sum(len(p.split()) for p in paragraphs)
    filler_words_needed = max(0, target_words - para_words)

    # Distribute filler into (n+1) gaps: before first, between each pair, after last
    n_gaps = len(paragraphs) + 1
    words_per_gap = filler_words_needed // n_gaps

    parts = []
    for para in paragraphs:
        if words_per_gap > 0:
            parts.append(_generate_filler(words_per_gap, rng))
        parts.append(para)
    if words_per_gap > 0:
        parts.append(_generate_filler(words_per_gap, rng))

    return "\n\n".join(parts)


def _generate_filler(target_words: int, rng: random.Random) -> str:
    """Generate filler text of approximately target_words length."""
    sentences = []
    count = 0
    while count < target_words:
        sent = rng.choice(_FILLER_SENTENCES)
        sentences.append(sent)
        count += len(sent.split())
    return " ".join(sentences)


def load_musique(
    max_samples: int | None = None,
    doc_lengths: list[int] | None = None,
    hops_list: list[int] | None = None,
) -> list[MuSiQueSample]:
    """Load MuSiQue benchmark samples with padded documents.

    For each combination of hop count and doc_length, select up to max_samples
    raw questions and pad each document to the target length.
    """
    cfg = settings.benchmark_cfg.get("musique", {})
    n = max_samples or cfg.get("num_samples", 15)
    lengths = doc_lengths or cfg.get("doc_lengths", [10000, 50000])
    hops = hops_list or cfg.get("hops", [2, 3, 4])

    raw = _load_musique_raw()
    rng = random.Random(settings.seed)

    # Group raw samples by hop count
    by_hops: dict[int, list[dict]] = {}
    for item in raw:
        decomp = item.get("question_decomposition", [])
        h = _count_hops(decomp)
        by_hops.setdefault(h, []).append(item)

    samples = []
    for hop_count in hops:
        available = by_hops.get(hop_count, [])
        if not available:
            logger.warning("No MuSiQue samples with %d hops", hop_count)
            continue

        # Deterministic shuffle then take up to n
        hop_rng = random.Random(settings.seed + hop_count)
        selected = list(available)
        hop_rng.shuffle(selected)
        selected = selected[:n]

        for doc_len in lengths:
            for i, item in enumerate(selected):
                paragraphs = _get_paragraphs(item)
                decomp = item.get("question_decomposition", [])

                pad_rng = random.Random(settings.seed + hop_count * 10000 + doc_len + i)
                document = _pad_document(paragraphs, doc_len, pad_rng)

                sid = f"musique_{hop_count}hop_{doc_len}w_{i}"
                samples.append(MuSiQueSample(
                    document=document,
                    question=item["question"],
                    answer=item["answer"],
                    answer_aliases=item.get("answer_aliases", []),
                    hops=hop_count,
                    bridge_entities=_extract_bridge_entities(decomp),
                    sub_questions=_extract_sub_questions(decomp),
                    doc_length=doc_len,
                    sample_id=sid,
                ))

    logger.info(
        "Loaded %d MuSiQue samples (%d hops x %d lengths x up to %d each)",
        len(samples), len(hops), len(lengths), n,
    )
    return samples
