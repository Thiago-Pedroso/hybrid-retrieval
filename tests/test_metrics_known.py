import numpy as np
from src.eval.metrics import mrr_at_k, ndcg_at_k, average_precision_at_k, recall_at_k, precision_at_k

def test_metrics_known_small_case():
    # ranked: 5 docs, relevantes: {d2, d4}, ganhos binários
    ranked = ["d1","d2","d3","d4","d5"]
    gold = {"d2","d4"}
    gains = {d: 1 for d in gold}

    assert abs(mrr_at_k(ranked, gold, k=10) - 1/2) < 1e-9  # d2 é o 2º
    # nDCG@3: ideal@3 = [1,1,0] => IDCG = 1 + 1/log2(3) = 1 + 1/1.58496...
    # DCG@3: [0,1,0] => 1/log2(3)
    idcg = (2**1 - 1) / np.log2(1+1) + (2**1 - 1) / np.log2(2+1) + 0
    dcg  = 0 + (2**1 - 1) / np.log2(2+1) + 0
    assert abs(ndcg_at_k(ranked, gains, k=3) - (dcg/idcg)) < 1e-9

    # MAP@5: acertos nas posições 2 e 4 => (1/2 + 2/4) / 2 = (0.5 + 0.5)/2 = 0.5
    assert abs(average_precision_at_k(ranked, gold, k=5) - 0.5) < 1e-9

    # Recall@3: só d2 aparece no top3 => 1/2
    assert abs(recall_at_k(ranked, gold, k=3) - 0.5) < 1e-9

    # Precision@3: 1 relevante / 3 = 0.3333
    assert abs(precision_at_k(ranked, gold, k=3) - (1/3)) < 1e-9
