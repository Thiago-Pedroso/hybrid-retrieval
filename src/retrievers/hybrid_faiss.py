from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from ..datasets.schema import Document, Query
from ..tri_modal.vectorizer import TriModalVectorizer
from ..tri_modal.hybrid_index import HybridIndex
from ..tri_modal.reranker import TriModalReranker
from ..tri_modal.weighting import StaticPolicy, HeuristicLLMPolicy
from .base import AbstractRetriever

class HybridRetriever(AbstractRetriever):
    def __init__(self,
                 sem_dim: Optional[int] = None,
                 tfidf_dim: int = 1000,
                 topk_first: int = 150,
                 policy: str = "heuristic",
                 # SEMÂNTICO
                 semantic_backend: str = "hf",
                 semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 query_prefix: str = "",
                 doc_prefix: str = "",
                 # TF-IDF
                 tfidf_backend: str = "sklearn",
                 # ENTIDADES
                 graph_model_name: str = "BAAI/bge-large-en-v1.5",
                 ner_backend: str = "scispacy",
                 ner_model: Optional[str] = None,
                 ner_use_noun_chunks: bool = True,
                 ner_batch_size: int = 64,
                 ner_n_process: int = 1,
                 # DISPOSITIVO
                 device: Optional[str] = None):
        self.vec = TriModalVectorizer(
            sem_dim=sem_dim,
            tfidf_dim=tfidf_dim,
            semantic_backend=semantic_backend,
            semantic_model_name=semantic_model_name,
            tfidf_backend=tfidf_backend,
            query_prefix=query_prefix,
            doc_prefix=doc_prefix,
            graph_model_name=graph_model_name,
            ner_backend=ner_backend,
            ner_model=ner_model,
            ner_use_noun_chunks=ner_use_noun_chunks,
            ner_batch_size=ner_batch_size,
            ner_n_process=ner_n_process,
            device=device,
        )
        self.index = HybridIndex(self.vec)
        self.reranker = TriModalReranker(self.vec)
        self.topk_first = topk_first
        self.policy = HeuristicLLMPolicy() if policy == "heuristic" else StaticPolicy()
        self._doc_map: Dict[str, str] = {}  # doc_id -> texto

    def build_index(self, docs: List[Document]):
        # Fit com textos completos (title + text)
        self.vec.fit_corpus((d.title or "") + " " + (d.text or "") for d in docs)
        self.index.build((d.doc_id, (d.title or "") + " " + (d.text or "")) for d in docs)
        self._doc_map = {d.doc_id: (d.title or "") + " " + (d.text or "") for d in docs}

    def retrieve(self, queries: List[Query], k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        results: Dict[str, List[Tuple[str, float]]] = {}
        for q in queries:
            q_vec = self.vec.concat(self.vec.encode_text(q.text, is_query=True))
            # 1ª fase
            candidates = self.index.search(q_vec, topk=self.topk_first)
            cand_texts = [(doc_id, self._doc_map[doc_id]) for doc_id, _ in candidates]
            # pesos
            w = self.policy.weights(q.text)
            reranked = self.reranker.rescore(q.text, cand_texts, w)
            results[q.query_id] = reranked[:k]
        return results
