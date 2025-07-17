from abc import ABC, abstractmethod
from typing import List


class EmbeddingModel(ABC):
    """Abstract base class for all embedding models."""

    @abstractmethod
    def embed(self, sentences: List[str]) -> List[List[float]]:
        """Convert a list of sentences into their embedding vectors."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Return a human-readable name for the embedding model."""
        pass

    @abstractmethod
    def list_models(self) -> list[str]:
        pass
