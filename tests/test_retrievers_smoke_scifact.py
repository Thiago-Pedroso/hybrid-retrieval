from pathlib import Path
from src.datasets.loader import load_beir_dataset, select_split, as_documents, as_queries
from src.retrievers.bm25_basic import BM25Basic
from src.retrievers.dense_faiss import DenseFaiss
from src.retrievers.hybrid_faiss import HybridRetriever
from src.vectorizers.factory import create_vectorizer
from src.indexes.factory import create_index
from src.fusion.factory import create_weight_policy, create_reranker

def _prep_scifact_subset(n_docs=1000, n_queries=50):
    root = Path("./data/scifact/processed/beir").resolve()
    corpus, queries, qrels = load_beir_dataset(root)
    split = select_split(qrels, ("test","dev","validation","train"))
    qrels = qrels[qrels["split"] == split].copy()

    # reduz tamanho para rápida execução
    # pega só doc_ids referenciados + alguns extras
    used_docs = set(qrels["doc_id"].astype(str).unique().tolist())
    subset_corpus = corpus[corpus["doc_id"].astype(str).isin(used_docs)].head(n_docs)
    docs = as_documents(subset_corpus)

    qids = list(qrels["query_id"].astype(str).unique())[:n_queries]
    subset_queries = queries[queries["query_id"].isin(qids)]
    qlist = as_queries(subset_queries)
    return docs, qlist

def _assert_k(results, qlist, k):
    assert set(results.keys()) == set(q.query_id for q in qlist)
    for q in qlist:
        assert len(results[q.query_id]) <= k
        # ordenado desc por score
        scores = [s for _, s in results[q.query_id]]
        assert scores == sorted(scores, reverse=True)

def test_bm25_smoke():
    docs, qlist = _prep_scifact_subset()
    retr = BM25Basic()
    retr.build_index(docs)
    res = retr.retrieve(qlist, k=10)
    _assert_k(res, qlist, 10)

def test_dense_smoke():
    docs, qlist = _prep_scifact_subset()
    # DenseFaiss doesn't use dim parameter anymore, uses model directly
    retr = DenseFaiss(model_name="sentence-transformers/all-MiniLM-L6-v2")
    retr.build_index(docs)
    res = retr.retrieve(qlist, k=10)
    _assert_k(res, qlist, 10)

def test_hybrid_smoke():
    docs, qlist = _prep_scifact_subset()
    # Create components using new modular API
    vectorizer = create_vectorizer({
        "type": "tri_modal",
        "semantic": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        "tfidf": {"dim": 256},
        "graph": {"model": "BAAI/bge-large-en-v1.5"},
    })
    index = create_index({"type": "faiss", "metric": "ip"}, vectorizer)
    weight_policy = create_weight_policy({"policy": "heuristic"})
    reranker = create_reranker("tri_modal", vectorizer)
    retr = HybridRetriever(
        vectorizer=vectorizer,
        index=index,
        reranker=reranker,
        weight_policy=weight_policy,
        topk_first=50,
    )
    retr.build_index(docs)
    res = retr.retrieve(qlist, k=10)
    _assert_k(res, qlist, 10)
