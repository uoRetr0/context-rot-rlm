"""Abstract base controller for QA methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.environment.document_store import DocumentStore
from src.trace.tracer import TraceNode


@dataclass
class ControllerResult:
    answer: str
    confidence: float = 0.0
    method: str = ""
    trace: TraceNode | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseController(ABC):
    """Abstract base for all QA controllers."""

    method_name: str = "base"

    @abstractmethod
    def answer(
        self,
        question: str,
        store: DocumentStore,
        **kwargs: Any,
    ) -> ControllerResult:
        """Answer a question given a document store."""
        ...
