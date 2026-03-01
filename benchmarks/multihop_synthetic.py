"""Multi-hop synthetic question generator."""

from __future__ import annotations

import random
from dataclasses import dataclass

from src.config import settings

# Entity pools for generating connected facts
_PEOPLE = [
    "Dr. Amara Singh", "Professor Chen Wei", "Maria Gonzalez",
    "James Okafor", "Yuki Tanaka", "Fatima Al-Rashid",
    "Liam O'Brien", "Priya Patel", "Henrik Johansson",
    "Sofia Petrov", "Carlos Mendez", "Aisha Kamara",
]

_ORGANIZATIONS = [
    "Zenith Research Lab", "Meridian Institute", "Polaris Foundation",
    "Apex Technologies", "Nebula Consortium", "Vanguard Academy",
    "Horizon Corp", "Catalyst Group", "Prism Analytics",
]

_LOCATIONS = [
    "New Helsinki", "Port Azura", "Mount Crescendo",
    "Lake Serenity", "Ravensbrook", "Silverdale",
    "Thornfield", "Crystalport", "Windmere",
]

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
]


@dataclass
class MultihopSample:
    document: str
    question: str
    answer: str
    hops: int
    bridge_facts: list[str]


def _generate_2hop(rng: random.Random, doc_length: int, seed: int) -> MultihopSample:
    """Generate a 2-hop question requiring chaining two facts."""
    person = rng.choice(_PEOPLE)
    org = rng.choice(_ORGANIZATIONS)
    location = rng.choice(_LOCATIONS)
    year = rng.randint(2018, 2025)

    fact1 = f"{person} served as the lead researcher at {org} from {year} to {year + 2}."
    fact2 = f"The headquarters of {org} is located in {location}."

    question = f"Where was {person}'s workplace located when they were a lead researcher?"
    answer = location

    doc = _build_document([fact1, fact2], doc_length, rng)
    return MultihopSample(
        document=doc, question=question, answer=answer,
        hops=2, bridge_facts=[fact1, fact2],
    )


def _generate_3hop(rng: random.Random, doc_length: int, seed: int) -> MultihopSample:
    """Generate a 3-hop question requiring chaining three facts."""
    person1 = rng.choice(_PEOPLE)
    person2 = rng.choice([p for p in _PEOPLE if p != person1])
    org = rng.choice(_ORGANIZATIONS)
    location = rng.choice(_LOCATIONS)
    year = rng.randint(2018, 2024)

    fact1 = f"{person1} was mentored by {person2} during their doctoral studies."
    fact2 = f"{person2} founded {org} in {year}."
    fact3 = f"{org} operates its primary facility in {location}."

    question = f"Where does the organization founded by {person1}'s doctoral mentor operate its primary facility?"
    answer = location

    doc = _build_document([fact1, fact2, fact3], doc_length, rng)
    return MultihopSample(
        document=doc, question=question, answer=answer,
        hops=3, bridge_facts=[fact1, fact2, fact3],
    )


def _build_document(facts: list[str], target_words: int, rng: random.Random) -> str:
    """Embed facts in filler text at random positions."""
    filler_words = target_words - sum(len(f.split()) for f in facts)
    words_per_segment = filler_words // (len(facts) + 1)

    parts = []
    for fact in facts:
        filler = _generate_filler(words_per_segment, rng)
        parts.append(filler)
        parts.append(fact)
    parts.append(_generate_filler(words_per_segment, rng))

    return " ".join(parts)


def _generate_filler(target_words: int, rng: random.Random) -> str:
    sentences = []
    count = 0
    while count < target_words:
        sent = rng.choice(_FILLER_SENTENCES)
        sentences.append(sent)
        count += len(sent.split())
    return " ".join(sentences)


def generate_benchmark(
    num_samples: int | None = None,
    hops_list: list[int] | None = None,
    doc_length: int | None = None,
) -> list[MultihopSample]:
    """Generate the full multi-hop benchmark suite."""
    cfg = settings.benchmark_cfg["multihop"]
    n = num_samples or cfg["num_samples"]
    hops = hops_list or cfg["hops"]
    length = doc_length or cfg["doc_length"]

    samples = []
    rng = random.Random(settings.seed)

    for hop_count in hops:
        for i in range(n):
            seed = settings.seed + i + hop_count * 1000
            sample_rng = random.Random(seed)
            if hop_count == 2:
                sample = _generate_2hop(sample_rng, length, seed)
            else:
                sample = _generate_3hop(sample_rng, length, seed)
            samples.append(sample)

    return samples
