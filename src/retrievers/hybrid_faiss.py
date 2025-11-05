from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from ..datasets.schema import Document, Query
from ..core.interfaces import AbstractVectorizer, AbstractIndex, AbstractReranker, AbstractWeightPolicy
from ..vectorizers.tri_modal_vectorizer import TriModalVectorizer
from ..indexes.hybrid_index import HybridIndex
from ..fusion.reranker import TriModalReranker
from ..fusion.weighting import StaticPolicy, HeuristicLLMPolicy
from .base import AbstractRetriever
from ..utils.logging import get_logger, log_time, ProgressLogger

_log = get_logger("retriever.hybrid")

class HybridRetriever(AbstractRetriever):
    """Hybrid retriever that composes vectorizer, index, reranker, and weight policy."""
    
    def __init__(
        self,
        vectorizer: AbstractVectorizer,
        index: AbstractIndex,
        reranker: AbstractReranker,
        weight_policy: AbstractWeightPolicy,
        topk_first: int = 150,
    ):
        """Initialize HybridRetriever with modular components.
        
        Args:
            vectorizer: Vectorizer instance (must implement AbstractVectorizer)
            index: Index instance (must implement AbstractIndex)
            reranker: Reranker instance (must implement AbstractReranker)
            weight_policy: Weight policy instance (must implement AbstractWeightPolicy)
            topk_first: Number of candidates to retrieve before reranking
        """
        self.vec = vectorizer
        self.index = index
        self.reranker = reranker
        self.policy = weight_policy
        self.topk_first = topk_first
        self._doc_map: Dict[str, str] = {}  # doc_id -> texto

    def build_index(self, docs: List[Document]):
        _log.info(f"🚀 Building Hybrid Index para {len(docs)} documentos")
        
        # Fit com textos completos
        texts = [(d.title or "") + " " + (d.text or "") for d in docs]
        with log_time(_log, "Fit vectorizer no corpus"):
            self.vec.fit_corpus(texts)
        
        # Indexa
        doc_pairs = [(d.doc_id, (d.title or "") + " " + (d.text or "")) for d in docs]
        with log_time(_log, "Build search index"):
            self.index.build(doc_pairs)
        
        # Cache de textos para reranking
        _log.info("💾 Criando cache de textos para reranking...")
        self._doc_map = {d.doc_id: (d.title or "") + " " + (d.text or "") for d in docs}
        
        # Salva cache de embeddings de entidade (se configurado e disponível)
        try:
            if hasattr(self.vec, 'entities') and hasattr(self.vec.entities, 'save_embedding_cache'):
                with log_time(_log, "Salvando cache de embeddings"):
                    self.vec.entities.save_embedding_cache()
        except Exception as e:
            _log.debug(f"Cache de embeddings não salvo: {e}")
        
        _log.info(f"✅ Índice híbrido construído com sucesso!")

    def retrieve(self, queries: List[Query], k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        _log.info(f"🔍 Iniciando retrieval: {len(queries)} queries, top-{k} (topk_first={self.topk_first})")
        results: Dict[str, List[Tuple[str, float]]] = {}
        
        with ProgressLogger(_log, "Retrieval", total=len(queries), log_every=max(1, len(queries)//10)) as progress:
            for q in queries:
                # Encode query
                q_vec = self.vec.concat(self.vec.encode_text(q.text, is_query=True))
                
                # Fase 1: busca rápida
                candidates = self.index.search(q_vec, topk=self.topk_first)
                _log.debug(f"Query '{q.query_id[:20]}': {len(candidates)} candidatos da fase 1")
                
                # Fase 2: reranking com pesos adaptativos (se reranker disponível)
                if self.reranker is not None:
                    cand_texts = [(doc_id, self._doc_map[doc_id]) for doc_id, _ in candidates]
                    w = self.policy.weights(q.text)
                    _log.debug(f"  Pesos: {w}")
                    reranked = self.reranker.rescore(q.text, cand_texts, w)
                    results[q.query_id] = reranked[:k]
                else:
                    # Sem reranker, usa resultados diretos da busca
                    results[q.query_id] = candidates[:k]
                
                progress.update(1)
        
        _log.info(f"✅ Retrieval concluído: {len(results)} queries processadas")
        return results
