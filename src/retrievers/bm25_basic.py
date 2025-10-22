from __future__ import annotations
import math
import re
from typing import Dict, List, Tuple, Iterable, DefaultDict
from collections import defaultdict
from ..datasets.schema import Document, Query
from .base import AbstractRetriever

_tok = re.compile(r"[A-Za-z0-9_]+")

def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return [t.lower() for t in _tok.findall(text)]

class BM25Basic(AbstractRetriever):
    """Implementação enxuta de BM25 (k1,b) sobre título+texto concatenados."""
    def __init__(self, k1: float = 0.9, b: float = 0.4):
        self.k1 = k1
        self.b = b
        # index
        self.doc_ids: List[str] = []
        self.doc_lens: List[int] = []
        self.doc_texts: List[str] = []
        self.tf: List[DefaultDict[str, int]] = []
        self.df: DefaultDict[str, int] = defaultdict(int)
        self.avgdl: float = 0.0
        self.N: int = 0

    def build_index(self, docs: List[Document]) -> None:
        self.doc_ids, self.doc_lens, self.doc_texts, self.tf = [], [], [], []
        self.df.clear()
        for d in docs:
            txt = (d.title or "") + " " + (d.text or "")
            toks = tokenize(txt)
            self.doc_ids.append(d.doc_id)
            self.doc_texts.append(txt)
            self.doc_lens.append(len(toks))
            counts: DefaultDict[str, int] = defaultdict(int)
            for t in toks:
                counts[t] += 1
            self.tf.append(counts)
            for t in set(toks):
                self.df[t] += 1
        self.N = len(self.doc_ids)
        self.avgdl = (sum(self.doc_lens) / self.N) if self.N else 0.0

    def _score_query(self, q_toks: List[str], j: int) -> float:
        """Score BM25 de um doc j para uma query tokenizada."""
        score = 0.0
        dl = self.doc_lens[j] or 1
        for t in q_toks:
            f = self.df.get(t, 0)
            if f == 0:
                continue
            idf = math.log(1 + (self.N - f + 0.5) / (f + 0.5))
            tf = self.tf[j].get(t, 0)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
            score += idf * (tf * (self.k1 + 1)) / (denom or 1.0)
        return score

    def retrieve(self, queries: List[Query], k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        results: Dict[str, List[Tuple[str, float]]] = {}
        for q in queries:
            q_toks = tokenize(q.text)
            scores = [(self.doc_ids[j], self._score_query(q_toks, j)) for j in range(self.N)]
            scores.sort(key=lambda x: x[1], reverse=True)
            results[q.query_id] = scores[:k]
        return results
