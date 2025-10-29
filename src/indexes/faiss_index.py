from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
import numpy as np
from ..utils.logging import get_logger

_log = get_logger("faiss.index")

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False


class FaissFlatIPIndex:
    """
    Encapsula IndexFlatIP + persistência de IDs.
    - Salva/carrega índice FAISS e mapeamento de IDs
    - Interface minimalista: build_from_matrix, search
    """
    def __init__(self, artifact_dir: Optional[str], index_name: str):
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.index_name = index_name
        self.index = None
        self.doc_ids: List[str] = []
        if self.artifact_dir:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            self._index_path = self.artifact_dir / index_name
            self._ids_path = self.artifact_dir / f"{index_name}.ids.json"
        else:
            self._index_path = None
            self._ids_path = None

    def try_load(self) -> bool:
        if not (_HAS_FAISS and self._index_path and self._index_path.exists() and self._ids_path and self._ids_path.exists()):
            return False
        try:
            self.index = faiss.read_index(str(self._index_path))
            with open(self._ids_path, "r", encoding="utf-8") as f:
                self.doc_ids = json.load(f)
            return True
        except Exception as e:
            warnings.warn(f"[FaissFlatIPIndex] Falha ao carregar índice: {e}")
            return False

    def save(self):
        if not (_HAS_FAISS and self.index and self._index_path and self._ids_path):
            return
        try:
            faiss.write_index(self.index, str(self._index_path))
            with open(self._ids_path, "w", encoding="utf-8") as f:
                json.dump(self.doc_ids, f, ensure_ascii=False)
        except Exception as e:
            warnings.warn(f"[FaissFlatIPIndex] Falha ao salvar índice: {e}")

    def build_from_matrix(self, doc_ids: List[str], doc_mat: np.ndarray):
        self.doc_ids = list(doc_ids)
        if _HAS_FAISS:
            d = int(doc_mat.shape[1] if doc_mat is not None and len(doc_mat.shape) == 2 else 0)
            idx = faiss.IndexFlatIP(d)
            if doc_mat is not None and d > 0:
                idx.add(doc_mat.astype(np.float32))
            self.index = idx
        else:
            self.index = None

    def search(self, q: np.ndarray, topk: int) -> List[Tuple[str, float]]:
        q = q.reshape(1, -1).astype(np.float32)
        if _HAS_FAISS and self.index is not None:
            scores, idx = self.index.search(q, topk)
            idx = idx[0].tolist()
            scores = scores[0].tolist()
            return [(self.doc_ids[i], float(scores[j])) for j, i in enumerate(idx)]
        # fallback NumPy
        if len(self.doc_ids) == 0:
            return []
        warnings.warn("[FaissFlatIPIndex] FAISS indisponível, usando fallback NumPy")
        # Não temos doc_mat aqui (por design); classe que usar NumPy deve gerenciar direto
        raise RuntimeError("NumPy fallback requer acesso à matriz de documentos; não suportado aqui.")