"""
Core interfaces (ABCs) for all components in the retrieval system.

These interfaces define the contracts that all implementations must follow,
enabling modularity and easy swapping of components.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Iterable, Optional, Any
import numpy as np
from ..datasets.schema import Document, Query


class AbstractRetriever(ABC):
    """Interface for retrievers that can index documents and retrieve top-K results."""
    
    @abstractmethod
    def build_index(self, docs: List[Document]) -> None:
        """Build or update the internal index from documents.
        
        Args:
            docs: List of documents to index
        """
        ...
    
    @abstractmethod
    def retrieve(self, queries: List[Query], k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Retrieve top-K documents for each query.
        
        Args:
            queries: List of queries
            k: Number of top results to return per query
            
        Returns:
            Dictionary mapping query_id to list of (doc_id, score) tuples,
            sorted by score (descending)
        """
        ...


class AbstractVectorizer(ABC):
    """Interface for vectorizers that convert text to vectors."""
    
    @abstractmethod
    def fit_corpus(self, docs_texts: Iterable[str]) -> None:
        """Fit the vectorizer on a corpus of documents.
        
        Args:
            docs_texts: Iterable of document texts
        """
        ...
    
    @abstractmethod
    def encode_text(self, text: str, is_query: bool = False) -> Dict[str, np.ndarray]:
        """Encode a text into vector(s).
        
        Args:
            text: Text to encode
            is_query: Whether this is a query (vs document)
            
        Returns:
            Dictionary mapping slice names to numpy arrays (e.g., {"s": vec_s, "t": vec_t})
        """
        ...
    
    @abstractmethod
    def concat(self, parts: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate vector parts into a single vector.
        
        Args:
            parts: Dictionary of slice names to vectors
            
        Returns:
            Concatenated and normalized vector
        """
        ...
    
    @property
    @abstractmethod
    def total_dim(self) -> int:
        """Total dimension of the concatenated vector."""
        ...


class AbstractFusionStrategy(ABC):
    """Interface for fusion strategies that combine multiple retrieval results."""
    
    @abstractmethod
    def fuse(
        self,
        query: str,
        results_list: List[Dict[str, List[Tuple[str, float]]]],
        weights: Optional[List[float]] = None,
    ) -> List[Tuple[str, float]]:
        """Fuse multiple retrieval results into a single ranked list.
        
        Args:
            query: Query text
            results_list: List of retrieval results, each as {query_id: [(doc_id, score), ...]}
            weights: Optional weights for each result set (if None, equal weights)
            
        Returns:
            Fused ranked list of (doc_id, score) tuples
        """
        ...


class AbstractReranker(ABC):
    """Interface for rerankers that rescore candidate documents."""
    
    @abstractmethod
    def rescore(
        self,
        query_text: str,
        candidate_docs: List[Tuple[str, str]],
        weights: Optional[Tuple[float, ...]] = None,
    ) -> List[Tuple[str, float]]:
        """Rescore candidate documents for a query.
        
        Args:
            query_text: Query text
            candidate_docs: List of (doc_id, doc_text) tuples
            weights: Optional weights for different modalities
            
        Returns:
            List of (doc_id, score) tuples sorted by score (descending)
        """
        ...


class AbstractWeightPolicy(ABC):
    """Interface for weight policies that determine fusion weights."""
    
    @abstractmethod
    def weights(self, query_text: str) -> Tuple[float, ...]:
        """Get weights for fusion based on query characteristics.
        
        Args:
            query_text: Query text
            
        Returns:
            Tuple of weights (one per modality/component)
        """
        ...


class AbstractMetric(ABC):
    """Interface for evaluation metrics."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the metric."""
        ...
    
    @abstractmethod
    def compute(
        self,
        ranked: List[str],
        gold: Dict[str, float],
        k: int = 10,
    ) -> float:
        """Compute metric value for a single query.
        
        Args:
            ranked: Ranked list of document IDs
            gold: Dictionary mapping doc_id to relevance score (0 = not relevant)
            k: Cutoff for evaluation
            
        Returns:
            Metric value (typically 0.0 to 1.0)
        """
        ...


class AbstractOutputFormatter(ABC):
    """Interface for output formatters that serialize evaluation results."""
    
    @property
    @abstractmethod
    def format_name(self) -> str:
        """Name of the output format."""
        ...
    
    @abstractmethod
    def format(
        self,
        results: Any,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """Format evaluation results.
        
        Args:
            results: Results to format (typically a DataFrame or dict)
            output_path: Optional path to save formatted output
            
        Returns:
            Formatted string if output_path is None, otherwise None
        """
        ...


class AbstractIndex(ABC):
    """Interface for search indexes."""
    
    @abstractmethod
    def build(self, doc_id_and_text: Iterable[Tuple[str, str]]) -> None:
        """Build the index from documents.
        
        Args:
            doc_id_and_text: Iterable of (doc_id, text) tuples
        """
        ...
    
    @abstractmethod
    def search(self, query_vec: np.ndarray, topk: int = 10) -> List[Tuple[str, float]]:
        """Search for top-K similar documents.
        
        Args:
            query_vec: Query vector
            topk: Number of results to return
            
        Returns:
            List of (doc_id, score) tuples sorted by score (descending)
        """
        ...

