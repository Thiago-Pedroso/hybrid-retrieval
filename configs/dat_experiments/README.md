# Experimentos DAT (Dynamic Alpha Tuning)

Esta pasta contém os experimentos para replicação do paper "Rethinking Hybrid Retrieval: When Small Embeddings and LLM Re-ranking Beat Bigger Models".

## Dataset

Todos os experimentos usam o dataset **squad_small**.

## Estrutura dos Experimentos

### Baselines (Alpha Fixo)

1. **01_baseline_alpha_0.0_bm25_only.yaml** - BM25 apenas (alpha=0.0)
2. **02_baseline_alpha_0.3.yaml** - Alpha fixo 0.3 (30% Dense, 70% BM25)
3. **03_baseline_alpha_0.6.yaml** - Alpha fixo 0.6 (60% Dense, 40% BM25) - valor ótimo do paper
4. **04_baseline_alpha_1.0_dense_only.yaml** - Dense apenas (alpha=1.0)

### DAT (Dynamic Alpha Tuning)

5. **05_dat_hybrid_gpt4o_mini.yaml** - DAT com GPT-4o-mini como LLM judge
6. **06_dat_hybrid_gpt4o.yaml** - DAT com GPT-4o como LLM judge (mais preciso, mais caro)

## Como Executar

Cada arquivo YAML contém o comando completo no topo do arquivo. Exemplo:

```bash
python scripts/run_experiment.py --config configs/dat_experiments/01_baseline_alpha_0.0_bm25_only.yaml
```

## Ordem Recomendada de Execução

1. Primeiro, execute os baselines para estabelecer linha de base:
   - 01 (BM25 only)
   - 04 (Dense only)
   - 03 (Alpha 0.6 - ótimo do paper)
   - 02 (Alpha 0.3 - comparação)

2. Depois, execute os experimentos DAT:
   - 05 (GPT-4o-mini - mais rápido e barato)
   - 06 (GPT-4o - mais preciso, se necessário)

## Resultados

Os resultados serão salvos em: `./outputs/experiments/dat_experiments/`

Cada experimento gera:
- Arquivo CSV com métricas
- Arquivo JSON com resultados detalhados

## Notas

- O índice FAISS será construído na primeira execução e reutilizado nas subsequentes
- O cache do LLM judge será reutilizado entre execuções (baseado em query_id + doc_ids)
- Todos os experimentos usam o mesmo índice FAISS para garantir comparabilidade

