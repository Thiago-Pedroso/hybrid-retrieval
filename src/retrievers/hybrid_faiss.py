from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from ..datasets.schema import Document, Query
from ..vectorizers.tri_modal_vectorizer import TriModalVectorizer
from ..indexes.hybrid_index import HybridIndex
from ..fusion.reranker import TriModalReranker
from ..fusion.weighting import StaticPolicy, HeuristicLLMPolicy
from .base import AbstractRetriever
from ..utils.logging import get_logger, log_time, ProgressLogger

_log = get_logger("retriever.hybrid")

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
                 ner_allowed_labels: Optional[List[str]] = None,
                 entity_artifact_dir: Optional[str] = None,
                 entity_force_rebuild: bool = False,
                 # DISPOSITIVO
                 device: Optional[str] = None,
                 # FAISS
                 faiss_factory: Optional[str] = None,     # ex.: "OPQ64,IVF4096,PQ64x8"
                 faiss_metric: str = "ip",
                 faiss_nprobe: Optional[int] = None,
                 faiss_train_size: int = 0,
                 index_artifact_dir: Optional[str] = None,
                 index_name: str = "hybrid.index"):
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
            ner_allowed_labels=ner_allowed_labels,
            entity_artifact_dir=entity_artifact_dir,
            entity_force_rebuild=entity_force_rebuild,
            device=device,
        )
        self.index = HybridIndex(
            vectorizer=self.vec,
            faiss_factory=faiss_factory,
            faiss_metric=faiss_metric,
            faiss_nprobe=faiss_nprobe,
            faiss_train_size=faiss_train_size,
            artifact_dir=index_artifact_dir,
            index_name=index_name,
        )
        self.reranker = TriModalReranker(self.vec)
        self.topk_first = topk_first
        self.policy = HeuristicLLMPolicy() if policy == "heuristic" else StaticPolicy()
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
        
        # Salva cache de embeddings de entidade (se configurado)
        try:
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
                
                # Fase 2: reranking com pesos adaptativos
                cand_texts = [(doc_id, self._doc_map[doc_id]) for doc_id, _ in candidates]
                w = self.policy.weights(q.text)
                _log.debug(f"  Pesos: semantic={w[0]:.2f}, tfidf={w[1]:.2f}, entities={w[2]:.2f}")
                reranked = self.reranker.rescore(q.text, cand_texts, w)
                results[q.query_id] = reranked[:k]
                
                progress.update(1)
        
        _log.info(f"✅ Retrieval concluído: {len(results)} queries processadas")
        return results
