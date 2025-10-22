from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_jsonl(path: Path, rows: List[dict]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path: Path) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def predictions_to_jsonl(preds: Dict[str, List[Tuple[str, float]]]) -> List[dict]:
    rows = []
    for qid, items in preds.items():
        rows.append({
            "query_id": qid,
            "results": [{"doc_id": d, "score": float(s)} for d, s in items]
        })
    return rows

def jsonl_to_predictions(rows: List[dict]) -> Dict[str, List[Tuple[str, float]]]:
    out = {}
    for r in rows:
        qid = r["query_id"]
        res = [(it["doc_id"], float(it["score"])) for it in r.get("results", [])]
        out[qid] = res
    return out
