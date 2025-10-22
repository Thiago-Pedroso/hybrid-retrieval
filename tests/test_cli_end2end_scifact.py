import os
import sys
import subprocess
from pathlib import Path
import pandas as pd

def run(argv_list):
    """Executa usando o MESMO interpretador do pytest (sys.executable)."""
    print(">>", " ".join([sys.executable] + argv_list))
    out = subprocess.run([sys.executable] + argv_list, check=True,
                         capture_output=True, text=True)
    print(out.stdout)
    if out.stderr:
        print(out.stderr)
    return out

def test_cli_end2end_scifact():
    data_root = Path("./data/scifact/processed/beir").resolve()
    assert data_root.exists(), "SciFact BEIR processado é necessário para o smoke."

    pred_path = Path("./outputs/smoke_scifact_hybrid.jsonl").resolve()
    met_path  = Path("./outputs/smoke_scifact_hybrid_metrics.csv").resolve()

    os.environ["PYTHONPATH"] = os.getcwd()

    # run_retrieval
    run(["scripts/run_retrieval.py", "--dataset-root", str(data_root),
         "--retriever", "hybrid", "--k", "10", "--out", str(pred_path)])

    assert pred_path.exists(), "Predições não foram geradas."

    # evaluate
    run(["scripts/evaluate.py", "--dataset-root", str(data_root),
         "--predictions", str(pred_path), "--ks", "1,3,5,10", "--out", str(met_path)])

    assert met_path.exists(), "Métricas não foram geradas."
    df = pd.read_csv(met_path)
    for col in ["k","MRR","nDCG","MAP","Recall","Precision"]:
        assert col in df.columns
    for col in ["MRR","nDCG","MAP","Recall","Precision"]:
        assert (df[col] >= 0).all() and (df[col] <= 1).all()
