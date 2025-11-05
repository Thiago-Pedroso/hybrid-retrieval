from __future__ import annotations
from typing import Iterable, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import re
import json
import warnings
import hashlib
import threading

from .encoders import HFSemanticEncoder, l2norm

_spacy_lock = threading.Lock()

@dataclass
class NERConfig:
    backend: str = "scispacy"      # "scispacy" | "spacy" | "none"
    model: Optional[str] = None
    use_noun_chunks: bool = True
    batch_size: int = 128
    n_process: int = 4
    allowed_labels: Optional[List[str]] = None
    # EntityRuler opcional: por padrão desativado e sem padrões embutidos
    use_entity_ruler: bool = False
    ruler_patterns: Optional[List[Dict]] = None

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
        
        _log.info(f"🔄 Carregando modelo de embeddings: {graph_model_name}")
        self.embedder = HFSemanticEncoder(model_name=graph_model_name, device=device)
        self.dim = int(self.embedder.dim or 1024)
        _log.info(f"✓ Modelo carregado: dim={self.dim}, device={device or 'cpu'}")

        # assinatura do cache: modelo + dim
        self._emb_signature = _safe_name(f"{self.model_name}_{self.dim}")

        # NER
        self.ner_cfg = ner or NERConfig()
        self._nlp = self._load_nlp(self.ner_cfg)
        self._has_noun_chunks = self.ner_cfg.use_noun_chunks and hasattr(self._nlp, "create_pipe") if self._nlp else False

        # vocabulário e idf
        self.ent2idf: Dict[str, float] = {}
        self._fitted = False

        # cache de embeddings (em memória)
        self._emb_cache: Dict[str, np.ndarray] = {}
        
        # cache persistido em memória (carregado uma vez)
        self._emb_mat: Optional[np.ndarray] = None
        self._emap: Optional[Dict[str, int]] = None

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
                    disable = ["tagger","lemmatizer","textcat"]
                    if not cfg.use_noun_chunks:
                        disable.append("parser")
                    nlp = spacy.load(m, disable=disable)
                    break
                except Exception:
                    continue

        if nlp is None:
            candidates = [cfg.model] if cfg.model else []
            candidates += ["en_core_web_trf", "en_core_web_md", "en_core_web_sm"]
            for m in candidates:
                try:
                    disable = ["tagger","lemmatizer","textcat"]
                    if not cfg.use_noun_chunks:
                        disable.append("parser")
                    nlp = spacy.load(m, disable=disable)
                    break
                except Exception:
                    continue

        if nlp is None:
            warnings.warn("[EntityEncoder] Nenhum modelo spaCy carregado; usando extrator simples baseado em regex.")
            return nlp

        # Se noun_chunks for necessário e o parser não estiver ativo, tenta habilitar
        try:
            if cfg.use_noun_chunks and "parser" not in getattr(nlp, "pipe_names", []):
                nlp.enable_pipe("parser")
        except Exception:
            pass

        # EntityRuler opcional: apenas quando explicitamente solicitado e com padrões fornecidos
        try:
            if cfg.use_entity_ruler and cfg.ruler_patterns:
                before = "ner" if "ner" in getattr(nlp, "pipe_names", []) else None
                ruler = nlp.add_pipe("entity_ruler", before=before)
                ruler.add_patterns(list(cfg.ruler_patterns))
        except Exception as e:
            warnings.warn(f"[EntityEncoder] Falha ao configurar EntityRuler: {e}")

        return nlp

    _simple_tok = re.compile(r"[A-Za-z][A-Za-z0-9_\-/\.]{1,}")
    _STOP = set(["the", "this", "that", "these", "those", "and", "or", "in", "of", "for", "to", "with", "on", "by", "an", "a", "is", "are", "be"])

    def _extract_simple(self, text: str) -> List[str]:
        cands = []
        for tok in self._simple_tok.findall(text or ""):
            if (len(tok) >= 10) or (tok[:1].isupper()):
                cands.append(tok)
        normed = []
        for t in cands:
            t = re.sub(r"\s+", " ", t).strip()
            t = t.strip(".,;:()[]{}").lower()
            # Filtrar stopwords básicas
            if len(t) >= 2 and t not in self._STOP:
                normed.append(t)
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
        # Não incluir noun_chunks quando allowed_labels está ativo (para evitar ruído)
        use_chunks = self.ner_cfg.use_noun_chunks and not self.ner_cfg.allowed_labels
        if use_chunks and hasattr(doc, "noun_chunks"):
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

        return normed

    def _extract_entities_batch(self, texts: Iterable[str]) -> List[List[str]]:
        if self._nlp is None:
            return [self._extract_simple(t) for t in texts]
        out: List[List[str]] = []
        
        texts_list = list(texts)
        import platform
        is_macos = platform.system() == 'Darwin'
        use_multiprocessing = (not is_macos and 
                              self.ner_cfg.n_process > 1 and 
                              len(texts_list) > 1)
        
        # Preparar parâmetros do pipe
        pipe_kwargs = {
            'batch_size': self.ner_cfg.batch_size,
            'disable': []
        }
        if use_multiprocessing:
            pipe_kwargs['n_process'] = self.ner_cfg.n_process

        if is_macos:
            with _spacy_lock:
                for doc in self._nlp.pipe(texts_list, **pipe_kwargs):
                    out.append(self._extract_entities_from_doc(doc))
        else:
            for doc in self._nlp.pipe(texts_list, **pipe_kwargs):
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

        import logging
        _log = logging.getLogger("entity_encoder")
        
        CHUNK_SIZE = 2000
        total = len(texts)
        
        _log.info(f"📊 NER fit: processando {total} documentos em chunks de {CHUNK_SIZE}")
        _log.info(f"   Backend: {self.ner_cfg.backend}, Batch size: {self.ner_cfg.batch_size}")
        
        for chunk_start in range(0, total, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, total)
            chunk = texts[chunk_start:chunk_end]
            
            for ents in self._extract_entities_batch(chunk):
                N += 1
                for e in set(ents):
                    df[e] = df.get(e, 0) + 1
            
            if chunk_end % 1000 == 0 or chunk_end == total:
                _log.info(f"   Processados: {chunk_end}/{total} docs ({chunk_end*100//total}%)")

        kept = {e: c for e, c in df.items() if c >= self.min_df}
        self.ent2idf = {e: float(np.log((1 + N) / (1 + c)) + 1.0) for e, c in kept.items()}
        self._fitted = True
        _log.info(f"✓ NER fit concluído: {len(self.ent2idf)} entidades únicas (min_df={self.min_df})")
        self._try_save_idf()

    def _load_emb_cache_files(self):
        """Carrega arquivos de cache de embeddings uma vez em memória."""
        if self._emb_mat is not None and self._emap is not None:
            return  # Já carregado
        
        if self._emap_path and self._embnpy_path and self._emap_path.exists() and self._embnpy_path.exists():
            try:
                with open(self._emap_path, "r", encoding="utf-8") as f:
                    self._emap = json.load(f)
                self._emb_mat = np.load(self._embnpy_path)
            except Exception as e:
                warnings.warn(f"[EntityEncoder] Erro ao carregar cache de embeddings: {e}")
                self._emap = None
                self._emb_mat = None

    def _get_emb(self, ent: str) -> np.ndarray:
        v = self._emb_cache.get(ent)
        if v is not None:
            return v
        
        # Carrega cache de arquivos uma vez (se ainda não carregado)
        self._load_emb_cache_files()
        
        # Tenta usar cache em memória
        if self._emap is not None and self._emb_mat is not None and ent in self._emap:
            row = int(self._emap[ent])
            vv = self._emb_mat[row].astype(np.float32)
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

        # Ordenar entidades por score TF*IDF (relevância) e aplicar cap apenas aqui
        scored_entities = [(e, tf[e] * self.ent2idf[e]) for e in tf.keys()]
        scored_entities.sort(key=lambda x: x[1], reverse=True)
        
        # Aplicar max_entities_per_text apenas nas top-k entidades por relevância
        if self.max_entities_per_text > 0 and len(scored_entities) > self.max_entities_per_text:
            scored_entities = scored_entities[:self.max_entities_per_text]

        acc = np.zeros(self.dim, dtype=np.float32)
        for e, _ in scored_entities:
            # Recalcular peso: tf já está no score, então dividimos pelo idf para obter tf
            f = tf[e]
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