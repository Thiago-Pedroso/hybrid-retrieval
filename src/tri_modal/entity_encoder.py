from __future__ import annotations
from typing import Iterable, Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import re
import warnings

from .encoders import HFSemanticEncoder, _StubSemanticEncoder, l2norm

@dataclass
class NERConfig:
    backend: str = "scispacy"      # "scispacy" | "spacy"
    model: Optional[str] = None    # se None, escolhemos um default de acordo com o backend
    use_noun_chunks: bool = True   # complementar NER com noun_chunks
    batch_size: int = 64
    n_process: int = 1             # use >1 se tiver CPU sobrando

class EntityEncoderReal:
    """
    Extrai entidades com spaCy/scispaCy, calcula IDF no corpus e embute cada entidade
    com um encoder HF (default: BGE-Large). Agrega por TF-IDF e normaliza (L2).

    Fórmula (por documento/consulta):
      g(text) = L2( sum_e [ tf(e,text) * idf(e) * emb(e) ] )

    - fit(corpus_texts): coleta DF(e) e calcula idf(e) para o vocabulário de entidades.
    - encode_text(text): usa TF-IDF das entidades presentes + embeddings (com cache).
    """
    def __init__(self,
                 graph_model_name: str = "BAAI/bge-large-en-v1.5",
                 device: Optional[str] = None,
                 ner: Optional[NERConfig] = None,
                 min_df: int = 2,
                 max_entities_per_text: int = 128):
        self.min_df = min_df
        self.max_entities_per_text = max_entities_per_text

        # Encoder de entidade (HF real com fallback)
        try:
            self.embedder = HFSemanticEncoder(model_name=graph_model_name, device=device)
            self.dim = int(self.embedder.dim or 1024)
            self._is_stub = False
        except Exception:
            warnings.warn("[EntityEncoder] Falling back to stub embedding encoder.")
            self.embedder = _StubSemanticEncoder(dim=384)
            self.dim = self.embedder.dim
            self._is_stub = True

        # NER
        self.ner_cfg = ner or NERConfig()
        self._nlp = self._load_nlp(self.ner_cfg)
        self._has_noun_chunks = self.ner_cfg.use_noun_chunks and hasattr(self._nlp, "create_pipe")

        # vocabulário e idf
        self.ent2idf: Dict[str, float] = {}
        self._fitted = False

        # cache de embeddings de entidade (surface form -> vetor)
        self._emb_cache: Dict[str, np.ndarray] = {}

    # ------------------------ NER loading ------------------------

    def _load_nlp(self, cfg: NERConfig):
        import spacy
        model_name = cfg.model
        nlp = None

        if cfg.backend == "scispacy":
            # tentativas de modelos scispaCy em ordem de preferência
            candidates = [
                "en_ner_bc5cdr_md",         # doenças/químicos
                "en_ner_bionlp13cg_md",     # biomédico amplo
                "en_core_sci_md",           # sem NER, mas bom tokenizer/vectors
            ]
            if model_name:
                candidates = [model_name] + candidates
            for m in candidates:
                try:
                    nlp = spacy.load(m, disable=["tagger","parser","lemmatizer","textcat"])
                    break
                except Exception:
                    continue

        if nlp is None:
            # fallback para spaCy geral
            candidates = [model_name] if model_name else []
            candidates += ["en_core_web_trf", "en_core_web_md", "en_core_web_sm"]
            for m in candidates:
                try:
                    nlp = spacy.load(m, disable=["tagger","parser","lemmatizer","textcat"])
                    break
                except Exception:
                    continue

        if nlp is None:
            raise RuntimeError("Nenhum modelo spaCy/scispaCy disponível. "
                               "Instale, por ex.: pip install scispacy && "
                               "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz")

        return nlp

    # ------------------------ extração de entidades ------------------------

    def _extract_entities_from_doc(self, doc) -> List[str]:
        ents = []
        # NER
        if getattr(doc, "ents", None):
            for e in doc.ents:
                txt = (e.text or "").strip()
                if txt:
                    ents.append(txt)
        # noun_chunks como complemento (opcional)
        if self.ner_cfg.use_noun_chunks and hasattr(doc, "noun_chunks"):
            try:
                for chunk in doc.noun_chunks:
                    txt = (chunk.text or "").strip()
                    if txt:
                        ents.append(txt)
            except Exception:
                pass

        # normalização básica do surface form
        normed = []
        for t in ents:
            t = re.sub(r"\s+", " ", t).strip()
            t = t.strip(".,;:()[]{}").lower()
            if len(t) >= 2:
                normed.append(t)
        # limitar para não explodir
        if len(normed) > self.max_entities_per_text:
            normed = normed[: self.max_entities_per_text]
        return normed

    def _extract_entities_batch(self, texts: Iterable[str]) -> List[List[str]]:
        # usa nlp.pipe para velocidade
        out: List[List[str]] = []
        for doc in self._nlp.pipe(texts, batch_size=self.ner_cfg.batch_size, n_process=self.ner_cfg.n_process, disable=[]):
            out.append(self._extract_entities_from_doc(doc))
        return out

    # ------------------------ fit: calcula IDF ------------------------

    def fit(self, corpus_texts: Iterable[str]):
        df: Dict[str, int] = {}
        N = 0
        texts = list(corpus_texts)
        if not texts:
            self.ent2idf = {}
            self._fitted = True
            return

        for ents in self._extract_entities_batch(texts):
            N += 1
            for e in set(ents):
                df[e] = df.get(e, 0) + 1

        # mantém apenas entidades acima de min_df
        kept = {e: c for e, c in df.items() if c >= self.min_df}
        # idf suavizado
        self.ent2idf = {e: float(np.log((1 + N) / (1 + c)) + 1.0) for e, c in kept.items()}
        self._fitted = True

    # ------------------------ encode: TF-IDF * embed(entity) ------------------------

    def _get_emb(self, ent: str) -> np.ndarray:
        v = self._emb_cache.get(ent)
        if v is not None:
            return v
        # usamos o embedder de frase (HFSemanticEncoder). Para entidades curtas funciona bem.
        v = self.embedder.encode_text(ent, is_query=False)
        self._emb_cache[ent] = v
        return v

    def encode_text(self, text: str) -> np.ndarray:
        assert self._fitted, "Chame fit() antes de encode_text()."
        ents = self._extract_entities_batch([text or ""])[0]

        # TF por entidade
        tf: Dict[str, int] = {}
        for e in ents:
            if e in self.ent2idf:
                tf[e] = tf.get(e, 0) + 1
        if not tf:
            return np.zeros(self.dim, dtype=np.float32)

        # Agrega tf-idf * emb(e)
        acc = np.zeros(self.dim, dtype=np.float32)
        for e, f in tf.items():
            w = float(f) * self.ent2idf[e]
            acc += w * self._get_emb(e)

        # Normalização L2 final
        return l2norm(acc)
