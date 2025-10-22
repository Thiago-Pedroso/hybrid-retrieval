from __future__ import annotations
import re
import numpy as np
from typing import Iterable, List, Dict, Optional, Callable

# ========================= utilidades =========================

_tok = re.compile(r"[A-Za-z0-9_]+")

def _tokenize_basic(text: str) -> List[str]:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return [t.lower() for t in _tok.findall(text)]

def l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x) + 1e-12
    return x / n

# ========================= Semantic Encoders =========================

class _StubSemanticEncoder:
    """Stub determinístico (hash) para fallback."""
    def __init__(self, dim: int = 384, seed: int = 42, **kwargs):
        self.dim = dim
        self.seed = seed

    def _tok_vec(self, tok: str) -> np.ndarray:
        h = abs(hash((tok, self.seed))) % (2**32)
        _rng_local = np.random.default_rng(h)
        v = _rng_local.standard_normal(self.dim).astype(np.float32)
        return v

    def encode_text(self, text: str, is_query: bool = False) -> np.ndarray:
        toks = _tokenize_basic(text)
        if not toks:
            return np.zeros(self.dim, dtype=np.float32)
        acc = np.zeros(self.dim, dtype=np.float32)
        for t in toks[:128]:
            acc += self._tok_vec(t)
        return l2norm(acc)

class HFSemanticEncoder:
    """
    Encoder semântico real baseado em Hugging Face.
    Prioridade:
      1) sentence-transformers (mais simples/rápido)
      2) transformers (AutoModel) + mean pooling
    """
    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None,
                 max_length: int = 512,
                 normalize: bool = True,
                 query_prefix: str = "",
                 doc_prefix: str = "",
                 **kwargs):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.normalize = normalize
        self.query_prefix = query_prefix or ""
        self.doc_prefix = doc_prefix or ""

        self._backend = None  # "st" | "hf"
        self._model = None
        self._tokenizer = None
        self.dim = None

        # tenta sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name, device=device or "cpu")
            # detecta dim
            try:
                self.dim = int(self._model.get_sentence_embedding_dimension())
            except Exception:
                self.dim = int(self._model[0].word_embedding_dimension)  # fallback raro
            self._backend = "st"
        except Exception:
            # tenta transformers
            try:
                import torch
                from transformers import AutoTokenizer, AutoModel
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._model = AutoModel.from_pretrained(model_name)
                self._model.eval()
                if device:
                    self._model.to(device)
                self._torch = torch
                self.dim = int(self._model.config.hidden_size)
                self._backend = "hf"
            except Exception:
                # último recurso: stub
                self._backend = "stub"
                self._stub = _StubSemanticEncoder(dim=384)

    def _mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        return summed / counts

    def encode_text(self, text: str, is_query: bool = False) -> np.ndarray:
        if self._backend == "st":
            # sentence-transformers lida com normalização internamente dependendo do modelo,
            # mas normalizamos sempre por consistência com o paper.
            prefix = self.query_prefix if is_query else self.doc_prefix
            emb = self._model.encode(prefix + (text or ""), batch_size=1, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
            return l2norm(emb.astype(np.float32)) if self.normalize else emb.astype(np.float32)

        if self._backend == "hf":
            import torch
            prefix = self.query_prefix if is_query else self.doc_prefix
            txt = prefix + (text or "")
            toks = self._tokenizer(txt, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
            if self.device:
                toks = {k: v.to(self.device) for k, v in toks.items()}
            with torch.no_grad():
                out = self._model(**toks)
                if hasattr(out, "last_hidden_state"):
                    pooled = self._mean_pool(out.last_hidden_state, toks["attention_mask"])
                else:
                    pooled = out[0].mean(dim=1)
                vec = pooled.squeeze(0).cpu().numpy().astype(np.float32)
            return l2norm(vec) if self.normalize else vec

        # stub
        return self._stub.encode_text(text, is_query=is_query)

# ========================= TF-IDF Encoders =========================

class TfidfEncoder:
    """
    TF-IDF com backend:
      - 'sklearn' (default): TfidfVectorizer (max_features = dim, min_df etc.)
      - 'pyserini': usa o Lucene analyzer do Pyserini para tokenização,
                    mas calcula TF-IDF via sklearn (robusto e multiplataforma).
    """
    def __init__(self,
                 dim: int = 1000,
                 min_df: int = 2,
                 backend: str = "sklearn",
                 language: str = "english"):
        self.dim = dim
        self.min_df = min_df
        self.backend = backend
        self.language = language

        self._vectorizer = None
        self._analyzer_fn: Optional[Callable[[str], List[str]]] = None
        self._fitted = False

        if backend == "pyserini":
            try:
                from pyserini.analysis import Analyzer, get_lucene_analyzer
                _an = Analyzer(get_lucene_analyzer(stemmer=None, stopwords=True, lang=language))
                def analyze(text: str) -> List[str]:
                    return _an.analyze(text or "")
                self._analyzer_fn = analyze
            except Exception:
                print("[TfidfEncoder] Aviso: Pyserini não disponível. Caindo para 'sklearn'.")
                self.backend = "sklearn"

        if self.backend == "sklearn":
            # se nenhuma fun tokenizadora for definida, usamos a básica
            self._analyzer_fn = self._analyzer_fn or _tokenize_basic

    def fit(self, documents: Iterable[str]):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except Exception as e:
            raise RuntimeError("scikit-learn é necessário para TfidfEncoder (pip install scikit-learn)") from e

        # Vamos usar tokenizador custom quando backend=pyserini; caso contrário, token básico
        def _tok(text):
            return self._analyzer_fn(text)

        self._vectorizer = TfidfVectorizer(
            tokenizer=_tok,
            preprocessor=None,
            token_pattern=None,  # necessário quando passamos tokenizer
            lowercase=True,
            stop_words=None,     # já lidamos no analyzer
            min_df=self.min_df,
            max_features=self.dim,
            dtype=np.float32,
            norm=None,           # normalização L2 faremos nós
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False
        )
        docs = list(documents)
        if len(docs) == 0:
            # evita crash em dataset vazio
            self._vectorizer.fit([""])
        else:
            self._vectorizer.fit(docs)
        self._fitted = True

    @property
    def vocab_size(self) -> int:
        if not self._fitted:
            return 0
        return len(self._vectorizer.vocabulary_)

    def encode_text(self, text: str) -> np.ndarray:
        assert self._fitted, "Chame fit() antes de encode_text()"
        X = self._vectorizer.transform([text or ""])  # scipy sparse
        v = X.toarray().reshape(-1).astype(np.float32)
        if v.size == 0:
            return v
        return l2norm(v)
