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
    doc_length: int = 10000


def _org_alias(name: str) -> str:
    return "".join(word[0] for word in name.split() if word[0].isalnum()).upper()


def _generate_2hop(rng: random.Random, doc_length: int, seed: int) -> MultihopSample:
    """Generate a 2-hop question requiring chaining two facts."""
    person = rng.choice(_PEOPLE)
    org = rng.choice(_ORGANIZATIONS)
    location = rng.choice(_LOCATIONS)
    decoy_org = rng.choice([o for o in _ORGANIZATIONS if o != org])
    decoy_location = rng.choice([l for l in _LOCATIONS if l != location])
    satellite_location = rng.choice([l for l in _LOCATIONS if l not in {location, decoy_location}])
    org_alias = _org_alias(org)
    year = rng.randint(2018, 2025)

    fact1 = f"{person} served as the lead researcher at {org} from {year} to {year + 2}."
    fact2 = f"Internal briefs shorten {org} to {org_alias}. The headquarters of {org_alias} is located in {location}."
    distractors = [
        f"After that appointment, {person} briefly advised {decoy_org} on a separate project.",
        f"{decoy_org} maintains its headquarters in {decoy_location}.",
        f"{org} also runs a satellite field office in {satellite_location}, but its headquarters remain elsewhere.",
    ]

    question = f"Where was {person}'s workplace located when they were a lead researcher?"
    answer = location

    doc = _build_document([fact1, fact2], distractors, doc_length, rng)
    return MultihopSample(
        document=doc, question=question, answer=answer,
        hops=2, bridge_facts=[fact1, fact2], doc_length=doc_length,
    )


def _generate_3hop(rng: random.Random, doc_length: int, seed: int) -> MultihopSample:
    """Generate a 3-hop question requiring chaining three facts."""
    person1 = rng.choice(_PEOPLE)
    person2 = rng.choice([p for p in _PEOPLE if p != person1])
    decoy_mentor = rng.choice([p for p in _PEOPLE if p not in {person1, person2}])
    org = rng.choice(_ORGANIZATIONS)
    decoy_org = rng.choice([o for o in _ORGANIZATIONS if o != org])
    location = rng.choice(_LOCATIONS)
    decoy_location = rng.choice([l for l in _LOCATIONS if l != location])
    satellite_location = rng.choice([l for l in _LOCATIONS if l not in {location, decoy_location}])
    org_alias = _org_alias(org)
    year = rng.randint(2018, 2024)

    fact1 = f"{person1} was mentored by {person2} during their doctoral studies."
    fact2 = f"{person2} founded {org} in {year}; archival filings abbreviate the organization as {org_alias}."
    fact3 = f"{org_alias} operates its primary facility in {location}."
    distractors = [
        f"{person1} later collaborated with {decoy_mentor} on an unrelated workshop series.",
        f"{decoy_mentor} founded {decoy_org} in {year + 1}.",
        f"{decoy_org} operates its primary facility in {decoy_location}.",
        f"{org} also maintains a satellite lab in {satellite_location}, but that is not its primary facility.",
    ]

    question = f"Where does the organization founded by {person1}'s doctoral mentor operate its primary facility?"
    answer = location

    doc = _build_document([fact1, fact2, fact3], distractors, doc_length, rng)
    return MultihopSample(
        document=doc, question=question, answer=answer,
        hops=3, bridge_facts=[fact1, fact2, fact3], doc_length=doc_length,
    )


def _build_document(
    facts: list[str],
    distractors: list[str],
    target_words: int,
    rng: random.Random,
) -> str:
    """Spread supporting facts and distractors throughout a long filler document."""
    blocks = facts + distractors
    rng.shuffle(blocks)

    filler_words = max(0, target_words - sum(len(block.split()) for block in blocks))
    gap_count = len(blocks) + 1
    base_words = filler_words // gap_count
    remainder = filler_words % gap_count

    parts = []
    for idx in range(gap_count):
        target = base_words + (1 if idx < remainder else 0)
        filler = _generate_filler(target, rng)
        if filler:
            parts.append(filler)
        if idx < len(blocks):
            parts.append(blocks[idx])

    return " ".join(parts)


def _generate_filler(target_words: int, rng: random.Random) -> str:
    if target_words <= 0:
        return ""

    words: list[str] = []
    while len(words) < target_words:
        words.extend(rng.choice(_FILLER_SENTENCES).split())
    return " ".join(words[:target_words])


def generate_benchmark(
    num_samples: int | None = None,
    hops_list: list[int] | None = None,
    doc_lengths: list[int] | None = None,
) -> list[MultihopSample]:
    """Generate the full multi-hop benchmark suite."""
    cfg = settings.benchmark_cfg["multihop"]
    n = num_samples or cfg["num_samples"]
    hops = hops_list or cfg["hops"]
    # Support both plural doc_lengths and legacy singular doc_length
    lengths = doc_lengths or cfg.get("doc_lengths", [cfg.get("doc_length", 10000)])

    samples = []

    for hop_count in hops:
        for length in lengths:
            for i in range(n):
                seed = settings.seed + i + hop_count * 1000 + length
                sample_rng = random.Random(seed)
                if hop_count == 2:
                    sample = _generate_2hop(sample_rng, length, seed)
                else:
                    sample = _generate_3hop(sample_rng, length, seed)
                samples.append(sample)

    return samples
