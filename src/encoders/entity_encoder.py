from __future__ import annotations
from typing import Iterable, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import re
import json
import warnings
import hashlib

from .encoders import HFSemanticEncoder, _StubSemanticEncoder, l2norm

@dataclass
class NERConfig:
    backend: str = "scispacy"      # "scispacy" | "spacy" | "none"
    model: Optional[str] = None
    use_noun_chunks: bool = True
    batch_size: int = 64
    n_process: int = 1
    allowed_labels: Optional[List[str]] = None

@dataclass
class CacheConfig:
    artifact_dir: Optional[Path] = None
    force_rebuild: bool = False

def _safe_name(s: str) -> str:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return f"{s}_{h}"

class EntityEncoderReal:
    """
    Extrai entidades (spaCy/scispaCy), calcula IDF no corpus e embute cada entidade
    com um encoder HF (default: BGE-Large). Agrega por TF-IDF e normaliza (L2).

    g(text) = L2( sum_e [ tf(e,text) * idf(e) * emb(e) ] )

    Cache persistente (versionado por modelo+dim):
      - entity_idf.json
      - entity_emb_{signature}.npy
      - entity_emb_map_{signature}.json
    """
    def __init__(self,
                 graph_model_name: str = "BAAI/bge-large-en-v1.5",
                 device: Optional[str] = None,
                 ner: Optional[NERConfig] = None,
                 min_df: int = 2,
                 max_entities_per_text: int = 128,
                 cache: Optional[CacheConfig] = None):
        self.min_df = min_df
        self.max_entities_per_text = max_entities_per_text
        self.cache = cache or CacheConfig()

        # Encoder de entidade (HF real com fallback)
        self.model_name = graph_model_name
        import logging
        _log = logging.getLogger("entity_encoder.init")
        
        try:
            _log.info(f"🔄 Carregando modelo de embeddings: {graph_model_name}")
            self.embedder = HFSemanticEncoder(model_name=graph_model_name, device=device)
            self.dim = int(self.embedder.dim or 1024)
            self._is_stub = False
            _log.info(f"✓ Modelo carregado: dim={self.dim}, device={device or 'cpu'}")
        except Exception as e:
            _log.error(f"❌ Falha ao carregar {graph_model_name}: {type(e).__name__}: {e}")
            _log.error("⚠️  Usando stub encoder (384d) - RESULTADOS PODEM SER DIFERENTES!")
            _log.error("    Instale: pip install sentence-transformers transformers torch")
            warnings.warn(f"[EntityEncoder] Falling back to stub: {e}")
            self.embedder = _StubSemanticEncoder(dim=384)
            self.dim = self.embedder.dim
            self._is_stub = True

        # assinatura do cache: modelo + dim
        self._emb_signature = _safe_name(f"{self.model_name}_{self.dim}")
        
        # VALIDAÇÃO CRÍTICA: Se usou stub mas esperava modelo real, PARE!
        if self._is_stub and "stub" not in graph_model_name.lower():
            raise RuntimeError(
                f"\n{'='*80}\n"
                f"ERRO CRÍTICO: Modelo '{graph_model_name}' não pôde ser carregado!\n"
                f"Está usando stub (384d) ao invés do modelo real (1024d).\n\n"
                f"CAUSA PROVÁVEL:\n"
                f"  1. Biblioteca não instalada: pip install sentence-transformers transformers torch\n"
                f"  2. Modelo não encontrado localmente e sem internet\n"
                f"  3. Erro de inicialização (veja logs acima)\n\n"
                f"SOLUÇÃO:\n"
                f"  pip install sentence-transformers transformers torch\n"
                f"  ou use --graph-model 'stub' se quiser usar o fallback intencionalmente\n"
                f"{'='*80}\n"
            )

        # NER
        self.ner_cfg = ner or NERConfig()
        self._nlp = self._load_nlp(self.ner_cfg)
        self._has_noun_chunks = self.ner_cfg.use_noun_chunks and hasattr(self._nlp, "create_pipe") if self._nlp else False

        # vocabulário e idf
        self.ent2idf: Dict[str, float] = {}
        self._fitted = False

        # cache de embeddings (em memória)
        self._emb_cache: Dict[str, np.ndarray] = {}

        # persistência
        self._idf_path = None
        self._emap_path = None
        self._embnpy_path = None
        if self.cache.artifact_dir is not None:
            self.cache.artifact_dir.mkdir(parents=True, exist_ok=True)
            self._idf_path   = self.cache.artifact_dir / "entity_idf.json"
            # arquivos versionados por assinatura
            self._emap_path  = self.cache.artifact_dir / f"entity_emb_map_{self._emb_signature}.json"
            self._embnpy_path= self.cache.artifact_dir / f"entity_emb_{self._emb_signature}.npy"
            
            # Log informativo sobre cache
            import logging
            _log = logging.getLogger("entity_encoder")
            _log.debug(f"EntityEncoder cache signature: {self._emb_signature}")
            _log.debug(f"  Model: {self.model_name}, Dim: {self.dim}")
            _log.debug(f"  Cache dir: {self.cache.artifact_dir}")
            
            if self.cache.force_rebuild:
                _log.info("🔄 Force rebuild habilitado - cache será regenerado")
                # Limpa arquivos antigos de cache de embeddings se existirem
                if self._emap_path.exists():
                    self._emap_path.unlink()
                    _log.debug(f"  Removido: {self._emap_path.name}")
                if self._embnpy_path.exists():
                    self._embnpy_path.unlink()
                    _log.debug(f"  Removido: {self._embnpy_path.name}")
            else:
                # Limpa cache de OUTRAS assinaturas (dimensões incompatíveis)
                if self.cache.artifact_dir:
                    import glob
                    other_caches = []
                    for pattern in ["entity_emb_map_*.json", "entity_emb_*.npy"]:
                        for f in glob.glob(str(self.cache.artifact_dir / pattern)):
                            fp = Path(f)
                            # Se não é o cache atual, remove
                            if self._emb_signature not in fp.name:
                                other_caches.append(fp)
                    
                    if other_caches:
                        _log.warning(f"⚠️  Encontrados {len(other_caches)} caches de outras assinaturas")
                        _log.warning("   Removendo para evitar conflitos de dimensão...")
                        for fp in other_caches:
                            try:
                                fp.unlink()
                                _log.debug(f"  Removido: {fp.name}")
                            except Exception as e:
                                _log.debug(f"  Erro ao remover {fp.name}: {e}")
                
                self._try_load_cache()
                if self._emb_cache:
                    _log.debug(f"✓ Cache carregado: {len(self._emb_cache)} embeddings em memória")

    def _load_nlp(self, cfg: NERConfig):
        if cfg.backend == "none":
            return None
        try:
            import spacy
        except Exception:
            warnings.warn("[EntityEncoder] spaCy não instalado; caindo para extrator simples.")
            return None
        
        nlp = None
        if cfg.backend == "scispacy":
            candidates = [cfg.model] if cfg.model else []
            candidates += ["en_ner_bc5cdr_md", "en_ner_bionlp13cg_md", "en_core_sci_md"]
            for m in candidates:
                try:
                    nlp = spacy.load(m, disable=["tagger","parser","lemmatizer","textcat"])
                    break
                except Exception:
                    continue

        if nlp is None:
            candidates = [cfg.model] if cfg.model else []
            candidates += ["en_core_web_trf", "en_core_web_md", "en_core_web_sm"]
            for m in candidates:
                try:
                    nlp = spacy.load(m, disable=["tagger","parser","lemmatizer","textcat"])
                    break
                except Exception:
                    continue

        if nlp is None:
            warnings.warn("[EntityEncoder] Nenhum modelo spaCy carregado; usando extrator simples baseado em regex.")
        return nlp

    _simple_tok = re.compile(r"[A-Za-z][A-Za-z0-9_\-/\.]{1,}")

    def _extract_simple(self, text: str) -> List[str]:
        cands = []
        for tok in self._simple_tok.findall(text or ""):
            if (len(tok) >= 10) or (tok[:1].isupper()):
                cands.append(tok)
        normed = []
        for t in cands:
            t = re.sub(r"\s+", " ", t).strip()
            t = t.strip(".,;:()[]{}").lower()
            if len(t) >= 2:
                normed.append(t)
        if len(normed) > self.max_entities_per_text:
            normed = normed[: self.max_entities_per_text]
        return normed

    def _extract_entities_from_doc(self, doc) -> List[str]:
        ents = []
        if getattr(doc, "ents", None):
            for e in doc.ents:
                if self.ner_cfg.allowed_labels and str(e.label_) not in set(self.ner_cfg.allowed_labels):
                    continue
                txt = (e.text or "").strip()
                if txt:
                    ents.append(txt)
        if self.ner_cfg.use_noun_chunks and hasattr(doc, "noun_chunks"):
            try:
                for chunk in doc.noun_chunks:
                    txt = (chunk.text or "").strip()
                    if txt:
                        ents.append(txt)
            except Exception:
                pass
        normed = []
        for t in ents:
            t = re.sub(r"\s+", " ", t).strip()
            t = t.strip(".,;:()[]{}").lower()
            if len(t) >= 2:
                normed.append(t)
        if len(normed) > self.max_entities_per_text:
            normed = normed[: self.max_entities_per_text]
        return normed

    def _extract_entities_batch(self, texts: Iterable[str]) -> List[List[str]]:
        if self._nlp is None:
            return [self._extract_simple(t) for t in texts]
        out: List[List[str]] = []
        
        # PROTEÇÃO Mac M1: Força n_process=1 para evitar segfault
        # Mac M1 com multiprocessing 'fork' + spaCy causa crash
        import platform
        n_proc = 1 if platform.system() == 'Darwin' else self.ner_cfg.n_process
        
        if n_proc != self.ner_cfg.n_process:
            import logging
            _log = logging.getLogger("entity_encoder")
            _log.warning(f"⚠️  Mac detectado: usando n_process=1 (era {self.ner_cfg.n_process}) para evitar segfault")
        
        for doc in self._nlp.pipe(texts, batch_size=self.ner_cfg.batch_size,
                                  n_process=n_proc, disable=[]):
            out.append(self._extract_entities_from_doc(doc))
        return out

    def fit(self, corpus_texts: Iterable[str]):
        if self._idf_path and self._idf_path.exists() and not self.cache.force_rebuild:
            self._fitted = True
            try:
                with open(self._idf_path, "r", encoding="utf-8") as f:
                    self.ent2idf = {k: float(v) for k, v in json.load(f).items()}
                import logging
                _log = logging.getLogger("entity_encoder")
                _log.info(f"✓ IDF carregado do cache: {len(self.ent2idf)} entidades")
            except Exception:
                pass
            return

        df: Dict[str, int] = {}
        N = 0
        texts = list(corpus_texts)
        if not texts:
            self.ent2idf = {}
            self._fitted = True
            return

        # === OTIMIZAÇÃO DE MEMÓRIA: Processa em chunks para Mac M1 8GB ===
        import gc
        import logging
        _log = logging.getLogger("entity_encoder")
        
        # Chunk size otimizado para 8GB RAM (SciFact ~5k docs)
        CHUNK_SIZE = 500  # Processa 500 docs por vez (~1.5GB RAM peak)
        total = len(texts)
        
        _log.info(f"📊 NER fit: processando {total} documentos em chunks de {CHUNK_SIZE}")
        _log.info(f"   Backend: {self.ner_cfg.backend}, Batch size: {self.ner_cfg.batch_size}")
        
        for chunk_start in range(0, total, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, total)
            chunk = texts[chunk_start:chunk_end]
            
            # Processa chunk
            for ents in self._extract_entities_batch(chunk):
                N += 1
                for e in set(ents):
                    df[e] = df.get(e, 0) + 1
            
            # Libera memória entre chunks (crítico para Mac M1 8GB)
            gc.collect()
            
            # Log de progresso a cada 1000 docs ou último chunk
            if chunk_end % 1000 == 0 or chunk_end == total:
                _log.info(f"   Processados: {chunk_end}/{total} docs ({chunk_end*100//total}%)")

        kept = {e: c for e, c in df.items() if c >= self.min_df}
        self.ent2idf = {e: float(np.log((1 + N) / (1 + c)) + 1.0) for e, c in kept.items()}
        self._fitted = True
        _log.info(f"✓ NER fit concluído: {len(self.ent2idf)} entidades únicas (min_df={self.min_df})")
        self._try_save_idf()

    def _get_emb(self, ent: str) -> np.ndarray:
        v = self._emb_cache.get(ent)
        if v is not None:
            return v
        # tenta carregar da matriz persistida CORRETA (versionada por assinatura)
        if self._emap_path and self._embnpy_path and self._emap_path.exists() and self._embnpy_path.exists():
            try:
                with open(self._emap_path, "r", encoding="utf-8") as f:
                    emap = json.load(f)
                if ent in emap:
                    row = int(emap[ent])
                    mat = np.load(self._embnpy_path)
                    vv = mat[row].astype(np.float32)
                    # VALIDAÇÃO ROBUSTA: checa dimensão
                    if vv.shape[0] == self.dim:
                        self._emb_cache[ent] = vv
                        return vv
                    else:
                        # Cache com dimensão errada - ignora e regenera
                        warnings.warn(
                            f"[EntityEncoder] Cache com dimensão incorreta para '{ent}': "
                            f"esperado {self.dim}, encontrado {vv.shape[0]}. "
                            f"Regenerando embedding (considere usar --entity-force-rebuild)."
                        )
            except Exception as e:
                warnings.warn(f"[EntityEncoder] Erro ao carregar cache para '{ent}': {e}")

        # embutir e cachear
        v = self.embedder.encode_text(ent, is_query=False)
        
        # VALIDAÇÃO CRÍTICA: verifica dimensão antes de cachear
        if v.shape[0] != self.dim:
            raise ValueError(
                f"[EntityEncoder] Embedding gerado com dimensão incorreta para '{ent}': "
                f"esperado {self.dim}, obtido {v.shape[0]}. "
                f"Modelo: {self.model_name}. Verifique a configuração do encoder."
            )
        
        self._emb_cache[ent] = v
        return v

    def encode_text(self, text: str) -> np.ndarray:
        assert self._fitted, "Chame fit() antes de encode_text()."
        ents = self._extract_entities_batch([text or ""])[0]

        tf: Dict[str, int] = {}
        for e in ents:
            if e in self.ent2idf:
                tf[e] = tf.get(e, 0) + 1
        if not tf:
            return np.zeros(self.dim, dtype=np.float32)

        acc = np.zeros(self.dim, dtype=np.float32)
        for e, f in tf.items():
            w = float(f) * self.ent2idf[e]
            acc += w * self._get_emb(e)
        return l2norm(acc)

    def _try_load_cache(self):
        try:
            if self._idf_path and self._idf_path.exists():
                with open(self._idf_path, "r", encoding="utf-8") as f:
                    self.ent2idf = {k: float(v) for k, v in json.load(f).items()}
                self._fitted = True
        except Exception:
            pass

    def _try_save_idf(self):
        if not self._idf_path:
            return
        try:
            with open(self._idf_path, "w", encoding="utf-8") as f:
                json.dump(self.ent2idf, f, ensure_ascii=False)
        except Exception as e:
            warnings.warn(f"[EntityEncoder] Falha ao salvar IDF: {e}")

    def save_embedding_cache(self):
        """Salva o cache de embeddings (versionado por modelo+dim)."""
        if not (self._emap_path and self._embnpy_path):
            return
        try:
            ents = list(self._emb_cache.keys())
            if not ents:
                return
            mat = np.stack([self._emb_cache[e] for e in ents], axis=0).astype(np.float32)
            np.save(self._embnpy_path, mat)
            emap = {e: i for i, e in enumerate(ents)}
            with open(self._emap_path, "w", encoding="utf-8") as f:
                json.dump(emap, f, ensure_ascii=False)
        except Exception as e:
            warnings.warn(f"[EntityEncoder] Falha ao salvar cache de embeddings: {e}")