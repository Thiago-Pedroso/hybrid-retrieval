"""
Teste end-to-end usando o novo sistema YAML.
Testa que o sistema completo funciona: carregar config, executar experimento, gerar métricas.
"""
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
    """Teste end-to-end usando YAML config."""
    data_root = Path("./data/scifact/processed/beir").resolve()
    assert data_root.exists(), "SciFact BEIR processado é necessário para o smoke."

    config_path = Path("./configs/test_hybrid_scifact.yaml").resolve()
    assert config_path.exists(), f"Config YAML não encontrado: {config_path}"
    
    # Output esperado do experimento
    output_dir = Path("./outputs").resolve()
    expected_csv = output_dir / "test_hybrid_scifact.csv"

    os.environ["PYTHONPATH"] = os.getcwd()

    # Executa experimento via YAML
    run(["scripts/run_experiment.py", "--config", str(config_path),
         "--override", f"dataset.root={data_root}"])

    # Verifica que o arquivo de resultados foi gerado
    assert expected_csv.exists(), f"Arquivo de métricas não foi gerado: {expected_csv}"

    # Verifica estrutura do CSV
    df = pd.read_csv(expected_csv)
    required_cols = ["k", "nDCG", "MRR", "MAP", "Recall", "Precision"]
    for col in required_cols:
        assert col in df.columns, f"Coluna '{col}' não encontrada no CSV"
    
    # Verifica valores válidos
    metric_cols = ["MRR", "nDCG", "MAP", "Recall", "Precision"]
    for col in metric_cols:
        assert (df[col] >= 0).all() and (df[col] <= 1).all(), \
            f"Valores de {col} fora do intervalo [0, 1]"
    
    # Verifica que há resultados para os k's esperados
    expected_ks = [1, 3, 5, 10]
    assert set(df["k"].unique()) == set(expected_ks), \
        f"k's esperados: {expected_ks}, encontrados: {df['k'].unique().tolist()}"
