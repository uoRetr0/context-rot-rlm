"""Needle-in-a-haystack benchmark generator for measuring context rot."""

from __future__ import annotations

import random
from dataclasses import dataclass

from src.config import settings

# Filler text pool (Paul Graham-style essays on varied topics)
_FILLER_SENTENCES = [
    "The development of software has always been an iterative process.",
    "Markets tend to reward those who can adapt to changing circumstances.",
    "Education systems around the world face similar challenges in preparing students.",
    "The history of technology shows that breakthrough innovations often come from unexpected directions.",
    "Urban planning requires balancing economic growth with quality of life.",
    "Scientific research depends on both rigorous methodology and creative thinking.",
    "The relationship between art and technology has evolved significantly.",
    "Environmental conservation requires cooperation across national boundaries.",
    "Financial markets reflect collective expectations about the future.",
    "Communication technology has fundamentally changed how people interact.",
    "Healthcare systems must balance accessibility with quality of care.",
    "Transportation infrastructure shapes the development of cities and regions.",
    "Agricultural practices have been transformed by modern technology.",
    "The media landscape continues to evolve with new platforms and formats.",
    "Legal systems adapt slowly to technological and social changes.",
    "Architecture reflects both cultural values and practical constraints.",
    "The energy sector faces unprecedented challenges and opportunities.",
    "Sports and athletics play an important role in society beyond entertainment.",
    "Manufacturing processes have been revolutionized by automation.",
    "The publishing industry has been transformed by digital technology.",
]

# Needle facts: unique, easily verifiable statements
_NEEDLE_FACTS = [
    {
        "needle": "The secret recipe for the legendary Zephyr cake requires exactly 7 grams of Madagascan vanilla, 3 tablespoons of moonflower honey, and must be baked at precisely 162 degrees Celsius for 47 minutes.",
        "question": "What temperature should the Zephyr cake be baked at and for how long?",
        "answer": "162 degrees Celsius for 47 minutes",
    },
    {
        "needle": "Professor Elara Nightingale of the Cerulean Institute discovered in 2019 that the migration pattern of Arctic terns follows a figure-eight route spanning exactly 71,000 kilometers annually.",
        "question": "How many kilometers do Arctic terns travel annually according to Professor Nightingale's discovery?",
        "answer": "71,000 kilometers",
    },
    {
        "needle": "The ancient city of Meridiana, buried beneath the Sahara Desert, was finally mapped using ground-penetrating radar in 2021, revealing 342 distinct buildings across 15 hectares.",
        "question": "How many distinct buildings were found in the ancient city of Meridiana?",
        "answer": "342 distinct buildings",
    },
    {
        "needle": "The Quantum Resonance Engine prototype, codenamed Project Helios, achieved a sustained output of 2.4 terawatts during its landmark 8-minute test on March 15, 2023.",
        "question": "What sustained output did Project Helios achieve during its test?",
        "answer": "2.4 terawatts",
    },
    {
        "needle": "The Verdant Protocol requires all participating nations to reduce industrial methane emissions by exactly 38 percent below 2020 levels by the year 2035.",
        "question": "By what percentage must participating nations reduce industrial methane emissions under the Verdant Protocol?",
        "answer": "38 percent",
    },
]


@dataclass
class NeedleSample:
    """A single needle-in-haystack test case."""

    document: str
    question: str
    answer: str
    needle_position: float  # 0.0 = start, 1.0 = end
    haystack_length: int  # total word count
    needle_text: str


def _generate_filler(target_words: int, rng: random.Random) -> str:
    """Generate filler text of approximately target_words length."""
    sentences = []
    word_count = 0
    while word_count < target_words:
        sent = rng.choice(_FILLER_SENTENCES)
        sentences.append(sent)
        word_count += len(sent.split())
    return " ".join(sentences)


def generate_needle_haystack(
    haystack_length: int = 10000,
    needle_position: float = 0.5,
    needle_idx: int = 0,
    seed: int | None = None,
) -> NeedleSample:
    """Generate a single needle-in-haystack sample."""
    rng = random.Random(seed)
    fact = _NEEDLE_FACTS[needle_idx % len(_NEEDLE_FACTS)]

    needle_words = len(fact["needle"].split())
    filler_words = haystack_length - needle_words
    insert_point = int(filler_words * needle_position)

    before = _generate_filler(insert_point, rng)
    after = _generate_filler(filler_words - insert_point, rng)

    document = before + " " + fact["needle"] + " " + after

    return NeedleSample(
        document=document,
        question=fact["question"],
        answer=fact["answer"],
        needle_position=needle_position,
        haystack_length=len(document.split()),
        needle_text=fact["needle"],
    )


def generate_benchmark(
    haystack_lengths: list[int] | None = None,
    needle_positions: list[float] | None = None,
    num_samples: int | None = None,
) -> list[NeedleSample]:
    """Generate the full needle-haystack benchmark suite."""
    cfg = settings.benchmark_cfg["needle_haystack"]
    lengths = haystack_lengths or cfg["haystack_lengths"]
    positions = needle_positions or cfg["needle_positions"]
    n = num_samples or cfg["num_samples"]

    samples = []
    for length in lengths:
        for pos in positions:
            for i in range(n):
                sample = generate_needle_haystack(
                    haystack_length=length,
                    needle_position=pos,
                    needle_idx=i,
                    seed=settings.seed + i + int(pos * 1000) + length,
                )
                samples.append(sample)

    return samples
