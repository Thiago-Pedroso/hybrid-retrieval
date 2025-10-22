from pathlib import Path
from src.datasets.loader import load_beir_dataset, select_split

def test_loader_scifact_contract():
    root = Path("./data/scifact/processed/beir").resolve()
    corpus, queries, qrels = load_beir_dataset(root)

    # colunas essenciais
    for col in ["doc_id","title","text"]:
        assert col in corpus.columns
    for col in ["query_id","query"]:
        assert col in queries.columns
    for col in ["query_id","doc_id","score","split"]:
        assert col in qrels.columns

    # integridade básica
    assert set(qrels["doc_id"].astype(str).unique()) <= set(corpus["doc_id"].astype(str).unique())
    assert set(qrels["query_id"].astype(str).unique()) <= set(queries["query_id"].astype(str).unique())

    # split preferível existente
    split = select_split(qrels)
    assert split in set(qrels["split"].unique().tolist())
