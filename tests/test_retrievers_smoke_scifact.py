from pathlib import Path
from src.datasets.loader import load_beir_dataset, select_split, as_documents, as_queries
from src.retrievers.bm25_basic import BM25Basic
from src.retrievers.dense_faiss import DenseFaissStub
from src.retrievers.hybrid_faiss import HybridRetriever

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
    retr = DenseFaissStub(dim=128)
    retr.build_index(docs)
    res = retr.retrieve(qlist, k=10)
    _assert_k(res, qlist, 10)

def test_hybrid_smoke():
    docs, qlist = _prep_scifact_subset()
    retr = HybridRetriever(tfidf_dim=256, topk_first=50, policy="heuristic")
    retr.build_index(docs)
    res = retr.retrieve(qlist, k=10)
    _assert_k(res, qlist, 10)
