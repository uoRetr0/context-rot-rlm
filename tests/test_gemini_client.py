"""Tests for Gemini JSON response parsing helpers."""

from src.gemini_client import _repair_json_object


def test_repair_json_object_recovers_answer_and_confidence_from_truncated_reasoning():
    raw = (
        '{\n'
        '  "answer": "Windmere",\n'
        '  "confidence": 0.9,\n'
        '  "reasoning": "The question asks where'
    )

    repaired = _repair_json_object(raw)

    assert repaired["answer"] == "Windmere"
    assert repaired["confidence"] == 0.9


def test_repair_json_object_recovers_boolean_and_arrays():
    raw = (
        '{'
        '"answer":"Ravensbrook",'
        '"confidence":0.7,'
        '"evidence_used":[3,5],'
        '"requires_multi_hop":true'
        '}'
    )

    repaired = _repair_json_object(raw)

    assert repaired["evidence_used"] == [3, 5]
    assert repaired["requires_multi_hop"] is True


def test_repair_json_object_recovers_repl_fields():
    raw = (
        '{'
        '"thought":"inspect the top result",'
        '"code":"hits = search(\\"alpha\\", top_k=1)\\nprint(hits)"'
        '}'
    )

    repaired = _repair_json_object(raw)

    assert repaired["thought"] == "inspect the top result"
    assert "search" in repaired["code"]


def test_repair_json_object_handles_invalid_backslash_apostrophe():
    raw = (
        '{'
        '"thought":"find O\\\'Brien",'
        '"code":"print(search(\\"Liam O\\\'Brien\\", top_k=1))"'
        '}'
    )

    repaired = _repair_json_object(raw)

    assert repaired["thought"] == "find O'Brien"
    assert "O'Brien" in repaired["code"]
