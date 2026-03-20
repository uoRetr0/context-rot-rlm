"""Needle-in-a-haystack benchmark generator for measuring context rot."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from src.config import settings

# Generic filler text pool.
_FILLER_SENTENCES = [
    "The development of software has always been an iterative process shaped by many tradeoffs.",
    "Markets tend to reward those who can adapt to changing circumstances over time.",
    "Education systems around the world face similar challenges in preparing students for uncertainty.",
    "The history of technology shows that breakthrough innovations often come from unexpected directions.",
    "Urban planning requires balancing economic growth with quality of life for residents.",
    "Scientific research depends on both rigorous methodology and creative thinking under constraints.",
    "The relationship between art and technology has evolved significantly over recent decades.",
    "Environmental conservation requires cooperation across national boundaries and institutions.",
    "Financial markets reflect collective expectations about the future as much as current facts.",
    "Communication technology has fundamentally changed how people interact at work and at home.",
    "Healthcare systems must balance accessibility with quality of care and financial sustainability.",
    "Transportation infrastructure shapes the development of cities and regions in lasting ways.",
    "Agricultural practices have been transformed by modern technology and data collection.",
    "The media landscape continues to evolve with new platforms and new business models.",
    "Legal systems adapt slowly to technological and social changes that happen quickly.",
    "Architecture reflects both cultural values and practical constraints in the built environment.",
    "The energy sector faces unprecedented challenges and opportunities during major transitions.",
    "Sports and athletics play an important role in society beyond entertainment alone.",
    "Manufacturing processes have been revolutionized by automation, robotics, and software.",
    "The publishing industry has been transformed by digital technology and mobile reading habits.",
]

# Needle facts with topic-specific distractors that create near-miss retrieval targets.
_NEEDLE_FACTS = [
    {
        "needle": "The secret recipe for the legendary Zephyr cake requires exactly 7 grams of Madagascan vanilla, 3 tablespoons of moonflower honey, and must be baked at precisely 162 degrees Celsius for 47 minutes.",
        "question": "What temperature should the Zephyr cake be baked at and for how long?",
        "answer": "162 degrees Celsius for 47 minutes",
        "topic_sentences": [
            "Zephyr cake notes are often archived beside other pastry experiments that use similar ingredient lists.",
            "Several bakers documented moonflower honey substitutions when trying to imitate the Zephyr cake.",
            "Recipe ledgers frequently mention temperature calibration problems in older Zephyr cake ovens.",
        ],
        "distractors": [
            "An imitation Zephyr cake recipe from a coastal bakery says the cake should be baked at 158 degrees Celsius for 44 minutes.",
            "A workshop handout for trainee bakers claims the Zephyr cake should be baked at 165 degrees Celsius for 47 minutes.",
            "One travel magazine version of the Zephyr cake says to bake it at 162 degrees Celsius for 52 minutes.",
            "A festival adaptation lists 160 degrees Celsius for 49 minutes as the best Zephyr cake schedule.",
            "A disputed Zephyr cake memo recommends 168 degrees Celsius for 41 minutes for a firmer texture.",
            "A cafe test batch recorded 162 degrees Celsius for 39 minutes in a smaller Zephyr cake pan.",
        ],
    },
    {
        "needle": "Professor Elara Nightingale of the Cerulean Institute discovered in 2019 that the migration pattern of Arctic terns follows a figure-eight route spanning exactly 71,000 kilometers annually.",
        "question": "How many kilometers do Arctic terns travel annually according to Professor Nightingale's discovery?",
        "answer": "71,000 kilometers",
        "topic_sentences": [
            "Cerulean Institute field reports often compare tern routes against earlier satellite estimates.",
            "Migration studies frequently disagree because some estimates count only one leg of the tern journey.",
            "Arctic tern tracking summaries often mix annual distance with one-season distance measurements.",
        ],
        "distractors": [
            "A prior Cerulean Institute briefing estimated the Arctic tern route at 68,000 kilometers annually.",
            "A comparative seabird report claimed the annual tern journey covered 74,000 kilometers.",
            "An older field guide rounded the Arctic tern migration pattern to 70,500 kilometers each year.",
            "A competing survey described a looping route of 63,000 kilometers annually for Arctic terns.",
            "A conference poster attributed to a graduate lab proposed an annual distance of 72,400 kilometers.",
            "One summary of the figure-eight pattern incorrectly listed the route as 69,800 kilometers annually.",
        ],
    },
    {
        "needle": "The ancient city of Meridiana, buried beneath the Sahara Desert, was finally mapped using ground-penetrating radar in 2021, revealing 342 distinct buildings across 15 hectares.",
        "question": "How many distinct buildings were found in the ancient city of Meridiana?",
        "answer": "342 distinct buildings",
        "topic_sentences": [
            "Archaeological summaries of Meridiana often separate building counts from total excavation features.",
            "Radar-based site maps commonly include overlapping signatures that can be mistaken for buildings.",
            "Field notes on Meridiana compare several competing building counts from early surveys.",
        ],
        "distractors": [
            "An early Meridiana radar draft suggested the city contained 324 distinct buildings.",
            "A desert archaeology review summarized the Meridiana site as having 348 buildings.",
            "One excavation overview reported 336 distinct buildings across the mapped area.",
            "A conference abstract described Meridiana as containing 352 built structures.",
            "A preliminary catalog listed 329 distinct buildings before the final radar pass.",
            "A teaching case study used a rounded estimate of about 340 buildings for Meridiana.",
        ],
    },
    {
        "needle": "The Quantum Resonance Engine prototype, codenamed Project Helios, achieved a sustained output of 2.4 terawatts during its landmark 8-minute test on March 15, 2023.",
        "question": "What sustained output did Project Helios achieve during its test?",
        "answer": "2.4 terawatts",
        "topic_sentences": [
            "Project Helios test notes often distinguish between peak output and sustained output figures.",
            "Prototype energy reports sometimes mix the 8-minute Helios run with shorter calibration bursts.",
            "Reviewers repeatedly warned that Helios summaries should not confuse headline output with average draw.",
        ],
        "distractors": [
            "A pre-release briefing for Project Helios cited a sustained output of 2.1 terawatts.",
            "One media recap incorrectly stated that Project Helios sustained 2.6 terawatts.",
            "A calibration report noted a short burst near 2.8 terawatts but not as the sustained test output.",
            "A disputed technical memo listed 2.3 terawatts as the landmark Helios result.",
            "An investor deck rounded the Helios test figure down to 2.2 terawatts.",
            "A summary of a separate subsystem run claimed a sustained output of 1.9 terawatts.",
        ],
    },
    {
        "needle": "The Verdant Protocol requires all participating nations to reduce industrial methane emissions by exactly 38 percent below 2020 levels by the year 2035.",
        "question": "By what percentage must participating nations reduce industrial methane emissions under the Verdant Protocol?",
        "answer": "38 percent",
        "topic_sentences": [
            "Climate agreements often differ on whether methane targets are measured against 2019 or 2020 baselines.",
            "Policy briefings on the Verdant Protocol compare mandatory cuts against several earlier draft targets.",
            "Industrial methane summaries frequently confuse the target year with the percentage reduction requirement.",
        ],
        "distractors": [
            "A draft of the Verdant Protocol proposed a 35 percent methane reduction target.",
            "A policy explainer summarized the Verdant target as 36 percent below 2020 levels.",
            "One regional annex referenced a 40 percent industrial methane cut for a stricter scenario.",
            "A conference handout described a 42 percent target that did not make the final protocol.",
            "A negotiation memo mentioned a fallback commitment of 33 percent below 2020 levels.",
            "A secondary summary incorrectly reported the Verdant Protocol threshold as 37 percent.",
        ],
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
    distractor_texts: list[str] = field(default_factory=list)
    difficulty: str = "standard"
    requested_position: float = 0.5


def _generate_filler(target_words: int, rng: random.Random, pool: list[str] | None = None) -> str:
    """Generate filler text of approximately target_words length."""
    pool = pool or _FILLER_SENTENCES
    sentences = []
    word_count = 0
    while word_count < target_words:
        sent = rng.choice(pool)
        sentences.append(sent)
        word_count += len(sent.split())
    return " ".join(sentences)


def _sample_distractors(fact: dict[str, object], count: int, rng: random.Random) -> list[str]:
    distractors = list(fact.get("distractors", []))
    if count <= 0 or not distractors:
        return []
    rng.shuffle(distractors)
    return distractors[: min(count, len(distractors))]


def _repeat_distractors(distractors: list[str], repeat_factor: int, rng: random.Random) -> list[str]:
    if repeat_factor <= 1 or not distractors:
        return list(distractors)

    repeated: list[str] = []
    for _ in range(repeat_factor):
        shuffled = list(distractors)
        rng.shuffle(shuffled)
        repeated.extend(shuffled)
    return repeated


def _build_confusion_blocks(distractors: list[str], repeats: int) -> list[str]:
    if not distractors or repeats <= 0:
        return []

    alternative_list = "; ".join(distractors)
    blocks = []
    templates = [
        "Archivists repeatedly argued over these conflicting alternatives: {alternatives}.",
        "Several unreliable summaries keep repeating the following candidate answers: {alternatives}.",
        "Cross-check notes warn that many near-miss alternatives appear in the archive, including: {alternatives}.",
    ]
    for i in range(repeats):
        blocks.append(templates[i % len(templates)].format(alternatives=alternative_list))
    return blocks


def _insert_distractors(text: str, distractors: list[str], rng: random.Random) -> str:
    if not distractors:
        return text

    sentences = [segment.strip() for segment in text.split(". ") if segment.strip()]
    if not sentences:
        return " ".join(distractors)

    for distractor in distractors:
        insert_at = rng.randrange(0, len(sentences) + 1)
        sentences.insert(insert_at, distractor)

    return ". ".join(sentences)


def generate_needle_haystack(
    haystack_length: int = 10000,
    needle_position: float = 0.5,
    needle_idx: int = 0,
    seed: int | None = None,
    difficulty: str = "standard",
    distractor_count: int = 0,
    distractor_repeat_factor: int = 1,
    confusion_block_repeats: int = 0,
) -> NeedleSample:
    """Generate a single needle-in-haystack sample."""
    rng = random.Random(seed)
    fact = _NEEDLE_FACTS[needle_idx % len(_NEEDLE_FACTS)]
    topical_pool = _FILLER_SENTENCES + list(fact.get("topic_sentences", []))

    selected_distractors = _sample_distractors(fact, distractor_count, rng)
    expanded_distractors = _repeat_distractors(selected_distractors, distractor_repeat_factor, rng)
    confusion_blocks = _build_confusion_blocks(selected_distractors, confusion_block_repeats)
    challenge_segments = expanded_distractors + confusion_blocks

    anchor_words = len(str(fact["needle"]).split()) + sum(len(text.split()) for text in challenge_segments)
    filler_words = max(haystack_length - anchor_words, 0)
    insert_point = int(filler_words * needle_position)

    filler_pool = topical_pool if difficulty in {"adversarial", "challenging"} else _FILLER_SENTENCES
    before = _generate_filler(insert_point, rng, filler_pool)
    after = _generate_filler(filler_words - insert_point, rng, filler_pool)

    if challenge_segments:
        if difficulty == "challenging":
            left_count = max(1, len(challenge_segments) // 2)
            before_cluster = " ".join(challenge_segments[:left_count])
            after_cluster = " ".join(challenge_segments[left_count:])
            before = " ".join(part for part in [before, before_cluster] if part)
            after = " ".join(part for part in [after_cluster, after] if part)
        else:
            left_count = len(challenge_segments) // 2
            before = _insert_distractors(before, challenge_segments[:left_count], rng)
            after = _insert_distractors(after, challenge_segments[left_count:], rng)

    document = " ".join(part for part in [before, str(fact["needle"]), after] if part)
    before_words = len(before.split())
    actual_position = before_words / max(len(document.split()) - len(str(fact["needle"]).split()), 1)

    return NeedleSample(
        document=document,
        question=str(fact["question"]),
        answer=str(fact["answer"]),
        needle_position=actual_position,
        haystack_length=len(document.split()),
        needle_text=str(fact["needle"]),
        distractor_texts=selected_distractors,
        difficulty=difficulty,
        requested_position=needle_position,
    )


def generate_benchmark(
    haystack_lengths: list[int] | None = None,
    needle_positions: list[float] | None = None,
    num_samples: int | None = None,
    *,
    difficulty: str = "standard",
    distractor_count: int | None = None,
) -> list[NeedleSample]:
    """Generate the full needle-haystack benchmark suite."""
    difficulty_to_cfg = {
        "standard": "needle_haystack",
        "adversarial": "needle_haystack_stress",
        "challenging": "needle_haystack_challenging",
    }
    cfg_name = difficulty_to_cfg.get(difficulty, "needle_haystack")
    if cfg_name not in settings.benchmark_cfg:
        cfg_name = "needle_haystack"
    cfg = settings.benchmark_cfg[cfg_name]
    lengths = haystack_lengths or cfg["haystack_lengths"]
    positions = needle_positions or cfg["needle_positions"]
    n = num_samples or cfg["num_samples"]
    distractors = distractor_count if distractor_count is not None else cfg.get("distractor_count", 0)
    repeat_factor = cfg.get("distractor_repeat_factor", 1)
    confusion_repeats = cfg.get("confusion_block_repeats", 0)

    samples = []
    for length in lengths:
        for pos in positions:
            for i in range(n):
                sample = generate_needle_haystack(
                    haystack_length=length,
                    needle_position=pos,
                    needle_idx=i,
                    seed=settings.seed + i + int(pos * 1000) + length,
                    difficulty=difficulty,
                    distractor_count=distractors,
                    distractor_repeat_factor=repeat_factor,
                    confusion_block_repeats=confusion_repeats,
                )
                samples.append(sample)

    return samples


def generate_stress_benchmark(
    haystack_lengths: list[int] | None = None,
    needle_positions: list[float] | None = None,
    num_samples: int | None = None,
    distractor_count: int | None = None,
) -> list[NeedleSample]:
    """Generate the harder, adversarial context-rot stress suite."""
    return generate_benchmark(
        haystack_lengths=haystack_lengths,
        needle_positions=needle_positions,
        num_samples=num_samples,
        difficulty="adversarial",
        distractor_count=distractor_count,
    )


def generate_challenging_benchmark(
    haystack_lengths: list[int] | None = None,
    needle_positions: list[float] | None = None,
    num_samples: int | None = None,
    distractor_count: int | None = None,
) -> list[NeedleSample]:
    """Generate a dense distractor-heavy challenge suite."""
    return generate_benchmark(
        haystack_lengths=haystack_lengths,
        needle_positions=needle_positions,
        num_samples=num_samples,
        difficulty="challenging",
        distractor_count=distractor_count,
    )
