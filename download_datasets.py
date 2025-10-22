"""
Downloader de datasets do paper (padrão reproduzível)
-----------------------------------------------------
Datasets cobertos:
- SciFact (original, via S3/AllenAI, e/ou via Hugging Face)
- SciFact no formato BEIR (via .zip oficial UKP)
- FIQA (formato BEIR, via .zip UKP)
- NFCorpus (formato BEIR, via .zip UKP)

Estrutura de pastas criada (usando --root ./data):
data/
  scifact/
    raw/
      original/     # S3/HF (claims_* e corpus.jsonl)
      beir/         # .zip + extraído (corpus.jsonl, queries.jsonl, qrels/*.tsv)
    processed/
      original/     # normalizado em parquet/jsonl
      beir/         # normalizado em parquet/jsonl
  fiqa/
    raw/beir/
    processed/beir/
  nfcorpus/
    raw/beir/
    processed/beir/

Requisitos:
- Python 3.9+
- pip install pandas pyarrow requests tqdm
  (opcionais para SciFact via HF: datasets)

Uso:
  python download_datasets.py --datasets scifact,fiqa,nfcorpus --root ./data --format parquet
"""

import argparse
import hashlib
import json
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from tqdm import tqdm

# -------------------- Dependências opcionais --------------------

def _lazy_import_datasets():
    try:
        from datasets import load_dataset
        return load_dataset
    except Exception:
        print("AVISO: 'datasets' não encontrado. SciFact via HF ficará indisponível. Use --scifact-source s3.", file=sys.stderr)
        raise

def _lazy_import_requests():
    try:
        import requests
        return requests
    except Exception:
        print("ERRO: 'requests' não encontrado. Instale com: pip install requests", file=sys.stderr)
        raise


# -------------------- Utilidades --------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _is_nested(x) -> bool:
    return isinstance(x, (dict, list, tuple, set))

def _to_json_string(x):
    try:
        if isinstance(x, (set, tuple)):
            x = list(x)
        return json.dumps(x, ensure_ascii=False, sort_keys=True)
    except Exception:
        return json.dumps(str(x), ensure_ascii=False)

def sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Converte colunas com estruturas aninhadas em strings JSON (evita erro do pyarrow)."""
    df = df.copy()
    for col in df.columns:
        col_values = df[col]
        has_nested = any(
            (v is not None and not (isinstance(v, float) and pd.isna(v)) and _is_nested(v))
            for v in col_values
        )
        if has_nested:
            df[col] = col_values.apply(lambda v: _to_json_string(v) if _is_nested(v) else v)
    return df

def save_df(df: pd.DataFrame, out: Path, fmt: str):
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df2 = sanitize_for_parquet(df)
        df2.to_parquet(out, index=False)
    elif fmt == "jsonl":
        df.to_json(out, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError(f"Formato não suportado: {fmt}")

def download_file(url: str, out_path: Path, chunk_size: int = 1 << 20):
    requests = _lazy_import_requests()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(out_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=f"Baixando {out_path.name}") as pbar:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    return out_path

def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def extract_tar_gz(tar_path: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Python 3.14+: filter="data" para segurança e compatibilidade
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=dest_dir, filter="data")

def extract_zip(zip_path: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def find_first(root: Path, filename: str) -> Optional[Path]:
    for p in root.rglob(filename):
        if p.is_file():
            return p
    return None

def find_all(root: Path, pattern: str) -> List[Path]:
    return [p for p in root.rglob(pattern) if p.is_file()]


# -------------------- Normalizações --------------------

# BEIR: corpus/queries/qrels
def normalize_beir_corpus_jsonl(jsonl_path: Path) -> pd.DataFrame:
    rows = read_jsonl(jsonl_path)
    recs = []
    for r in rows:
        recs.append({
            "doc_id": r.get("_id") or r.get("doc_id"),
            "title": r.get("title"),
            "text": r.get("text") or r.get("abstract"),
            "metadata": r.get("metadata"),
        })
    return pd.DataFrame.from_records(recs)

def normalize_beir_queries_jsonl(jsonl_path: Path) -> pd.DataFrame:
    rows = read_jsonl(jsonl_path)
    recs = []
    for r in rows:
        recs.append({"query_id": r.get("_id") or r.get("query_id"), "query": r.get("text")})
    return pd.DataFrame.from_records(recs)

def normalize_beir_qrels_tsv(tsv_path: Path, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")
    # Colunas esperadas: query-id, corpus-id, score
    # Harmoniza nomes alternativos:
    df = df.rename(columns={
        "query-id": "query_id", "query_id": "query_id", "qid": "query_id",
        "corpus-id": "doc_id", "corpus_id": "doc_id", "doc_id": "doc_id",
        "score": "score", "label": "score"
    })
    # Adiciona split
    df["split"] = split_name
    return df[["query_id", "doc_id", "score", "split"]]

# SciFact original
def normalize_scifact_corpus(jsonl_path: Path) -> pd.DataFrame:
    rows = read_jsonl(jsonl_path)
    norm = []
    for r in rows:
        norm.append({
            "doc_id": r.get("doc_id"),
            "title": r.get("title"),
            "text": r.get("abstract") or r.get("text"),
            "metadata": {k: v for k, v in r.items() if k not in {"doc_id", "title", "abstract", "text"}},
        })
    return pd.DataFrame(norm)

def normalize_scifact_claims(jsonl_path: Path, split_name: str) -> pd.DataFrame:
    rows = read_jsonl(jsonl_path)
    norm = []
    for r in rows:
        norm.append({
            "claim_id": r.get("id"),
            "split": split_name,
            "claim": r.get("claim"),
            "label": r.get("label"),
            "evidences": r.get("evidences"),
            "metadata": {k: v for k, v in r.items() if k not in {"id", "claim", "label", "evidences"}},
        })
    return pd.DataFrame(norm)


# -------------------- Downloaders --------------------

def download_scifact_original(root: Path, fmt: str, source: str = "s3"):
    """
    SciFact original (claims_{train,dev,test}.jsonl + corpus.jsonl)
    source: "s3" (tarball) ou "hf" (Hugging Face datasets: allenai/scifact)
    """
    ds_root = ensure_dir(root / "scifact")
    raw_dir = ensure_dir(ds_root / "raw" / "original")
    proc_dir = ensure_dir(ds_root / "processed" / "original")

    if source == "s3":
        url = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
        tar_path = raw_dir / "data.tar.gz"
        if not tar_path.exists():
            download_file(url, tar_path)
        extract_dir = raw_dir / "extracted"
        if not (extract_dir / "data").exists():
            extract_tar_gz(tar_path, extract_dir)
        data_dir = extract_dir / "data"

        corpus_path = data_dir / "corpus.jsonl"
        train_path = data_dir / "claims_train.jsonl"
        dev_path   = data_dir / "claims_dev.jsonl"
        test_path  = data_dir / "claims_test.jsonl"

    elif source == "hf":
        load_dataset = _lazy_import_datasets()
        claims = load_dataset("allenai/scifact", name="claims")
        corpus = load_dataset("allenai/scifact", name="corpus")

        corpus_jsonl = raw_dir / "corpus.jsonl"
        with open(corpus_jsonl, "w", encoding="utf-8") as f:
            for r in corpus["corpus"]:
                rec = {"doc_id": r.get("_id"), "title": r.get("title"), "abstract": r.get("abstract") or r.get("text")}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        train_jsonl = raw_dir / "claims_train.jsonl"
        dev_jsonl   = raw_dir / "claims_dev.jsonl"
        test_jsonl  = raw_dir / "claims_test.jsonl"

        def _dump_claim_split(split, outp: Path):
            with open(outp, "w", encoding="utf-8") as f:
                for r in claims[split]:
                    payload = {k: r.get(k) for k in r.keys()}
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        _dump_claim_split("train", train_jsonl)
        _dump_claim_split("validation", dev_jsonl)
        _dump_claim_split("test", test_jsonl)

        corpus_path = corpus_jsonl
        train_path = train_jsonl
        dev_path = dev_jsonl
        test_path = test_jsonl

    else:
        raise ValueError("--scifact-source deve ser 's3' ou 'hf'")

    # Normalizar -> processed/original
    if corpus_path.exists():
        df_corpus = normalize_scifact_corpus(corpus_path)
        save_df(df_corpus, proc_dir / f"corpus.{fmt}", fmt)
    if train_path.exists():
        df_train = normalize_scifact_claims(train_path, "train")
        save_df(df_train, proc_dir / f"claims_train.{fmt}", fmt)
    if dev_path.exists():
        df_dev = normalize_scifact_claims(dev_path, "dev")
        save_df(df_dev, proc_dir / f"claims_dev.{fmt}", fmt)
    if test_path.exists():
        df_test = normalize_scifact_claims(test_path, "test")
        save_df(df_test, proc_dir / f"claims_test.{fmt}", fmt)

    print(f"[OK] SciFact (original via {source}) salvo em {proc_dir}")

# MD5 oficiais (conforme wiki do BEIR) para verificação simples
BEIR_MD5 = {
    "nfcorpus": "a89dba18a62ef92f7d323ec890a0d38d",
    "fiqa":     "17918ed23cd04fb15047f73e6c3bd9d9",
    "scifact":  "5f7d1de60b170fc8027bb7898e2efca1",
}

def download_beir_zip_dataset(root: Path, name: str, fmt: str):
    """
    Baixa datasets do BEIR via .zip oficial (UKP):
      https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip
    Estrutura esperada após extração:
      {name}/
        corpus.jsonl
        queries.jsonl
        qrels/
          train.tsv | dev.tsv | test.tsv (presentes conforme dataset)
    """
    ds_root = ensure_dir(root / name)
    raw_dir = ensure_dir(ds_root / "raw" / "beir")
    proc_dir = ensure_dir(ds_root / "processed" / "beir")

    zip_url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
    zip_path = raw_dir / f"{name}.zip"
    if not zip_path.exists():
        download_file(zip_url, zip_path)

    expected = BEIR_MD5.get(name)
    if expected:
        got = md5sum(zip_path)
        if got != expected:
            print(f"[AVISO] MD5 de {zip_path.name} não confere! esperado={expected} obtido={got}. (Continuando mesmo assim)")

    extract_dir = raw_dir / "extracted"
    target_dir = extract_dir / name  # normalmente o zip cria a pasta com o nome do dataset
    if not target_dir.exists():
        extract_zip(zip_path, extract_dir)
    # fallback: se por algum motivo não existir {extract_dir}/{name}, procura arquivos
    base = target_dir if target_dir.exists() else extract_dir

    # Localiza arquivos
    corpus_path = find_first(base, "corpus.jsonl")
    queries_path = find_first(base, "queries.jsonl")
    qrels_dir = next((p for p in base.rglob("qrels") if p.is_dir()), None)
    qrels_files = []
    if qrels_dir:
        for p in qrels_dir.glob("*.tsv"):
            split = p.stem.lower()  # train/dev/test
            qrels_files.append((p, split))

    if not corpus_path or not queries_path or not qrels_files:
        raise RuntimeError(f"Estrutura inesperada em {name}. "
                           f"Encontrado corpus={bool(corpus_path)}, queries={bool(queries_path)}, qrels={len(qrels_files)}.")

    # Normaliza e salva
    df_corpus = normalize_beir_corpus_jsonl(corpus_path)
    df_queries = normalize_beir_queries_jsonl(queries_path)

    df_qrels_list = []
    for p, split in qrels_files:
        df_qrels_list.append(normalize_beir_qrels_tsv(p, split))
    df_qrels = pd.concat(df_qrels_list, ignore_index=True)

    save_df(df_corpus, proc_dir / f"corpus.{fmt}", fmt)
    save_df(df_queries, proc_dir / f"queries.{fmt}", fmt)
    save_df(df_qrels, proc_dir / f"qrels.{fmt}", fmt)

    print(f"[OK] BEIR/{name} salvo em {proc_dir}")


# -------------------- CLI --------------------

def parse_args():
    p = argparse.ArgumentParser(description="Downloader/normalizador para datasets do paper")
    p.add_argument("--datasets", type=str, required=True,
                   help="Lista separada por vírgulas: scifact,fiqa,nfcorpus (qualquer combinação)")
    p.add_argument("--root", type=str, default="./data", help="Diretório raiz para salvar datasets")
    p.add_argument("--format", type=str, choices=["parquet", "jsonl"], default="parquet", help="Formato de saída")
    p.add_argument("--scifact-source", type=str, choices=["s3", "hf"], default="s3",
                   help="Fonte do SciFact original: 's3' (tarball AllenAI) ou 'hf' (Hugging Face)")
    p.add_argument("--skip-beir", action="store_true", help="Pular downloads de BEIR (se o dataset suportar)")
    return p.parse_args()

def main():
    args = parse_args()
    root = Path(args.root).resolve()
    fmt = args.format
    ds_list = [x.strip().lower() for x in args.datasets.split(",") if x.strip()]

    for name in ds_list:
        if name == "scifact":
            # SciFact original
            download_scifact_original(root, fmt=fmt, source=args.scifact_source)
            # SciFact BEIR (via zip oficial), a menos que usuário peça para pular
            if not args.skip_beir:
                download_beir_zip_dataset(root, "scifact", fmt=fmt)
        elif name == "fiqa":
            if args.skip_beir:
                print("[AVISO] FIQA só está configurado via BEIR neste script. --skip-beir ignorado para 'fiqa'.")
            download_beir_zip_dataset(root, "fiqa", fmt=fmt)
        elif name == "nfcorpus":
            if args.skip_beir:
                print("[AVISO] NFCorpus só está configurado via BEIR neste script. --skip-beir ignorado para 'nfcorpus'.")
            download_beir_zip_dataset(root, "nfcorpus", fmt=fmt)
        else:
            print(f"[IGNORADO] Dataset desconhecido: {name}")

    print("\nConcluído. Estruturas em:", root)

if __name__ == "__main__":
    main()
