from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from ..datasets.schema import Document, Query

class AbstractRetriever(ABC):
    """Contrato mínimo: indexação idempotente e recuperação top-K."""

    @abstractmethod
    def build_index(self, docs: List[Document]) -> None:
        """Constroi/atualiza o índice interno (pode usar cache em impls futuras)."""
        ...

    @abstractmethod
    def retrieve(self, queries: List[Query], k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Retorna para cada query_id uma lista [(doc_id, score)] ordenada desc."""
        ...
