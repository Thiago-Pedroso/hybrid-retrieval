# Evaluation

Este módulo reúne as métricas e a rotina de avaliação para experimentos de Recuperação de Informação (IR) neste repositório.

Componentes principais:
- `metrics.py`: Implementações das métricas clássicas de IR (MRR, nDCG, MAP, Recall@k, Precision@k).
- `evaluator.py`: Função de avaliação que agrega métricas por `k` a partir de predições por consulta e dos `qrels` de referência.

## Formato esperado dos dados

- Predições (`preds`): dicionário no formato `{query_id: [(doc_id, score), ...]}` já ordenado por `score` (maior primeiro).
- Qrels (`qrels`): DataFrame com colunas `query_id`, `doc_id`, `score`, `split` (score > 0 indica relevância; pode ser binário ou ganho inteiro para nDCG).

## Métricas

- MRR@k (Mean Reciprocal Rank): média, sobre as consultas, do inverso da posição do primeiro documento relevante no ranking de tamanho `k`.
  - Intuição: "quão cedo aparece o primeiro relevante?". Varia entre 0 e 1.

- nDCG@k (Normalized Discounted Cumulative Gain): ganho acumulado descontado pela posição, normalizado pelo ganho ideal até `k`.
  - Intuição: considera vários documentos relevantes e seus graus de relevância, penalizando posições mais baixas. Varia entre 0 e 1.

- MAP@k (Mean Average Precision): média, por consulta, da precisão acumulada em cada acerto até `k`; depois, média entre consultas.
  - Intuição: recompensa rankings que colocam muitos relevantes cedo e de forma consistente.

- Recall@k: fração de documentos relevantes recuperados até `k`.
  - Intuição: "quanto do ouro total foi recuperado?". Varia entre 0 e 1.

- Precision@k: fração de documentos recuperados até `k` que são relevantes.
  - Intuição: "qual a pureza do top-k?". Varia entre 0 e 1.
  - **Nota**: Precision@k pode parecer baixa quando há poucos documentos relevantes por query.
    Por exemplo, se uma query tem apenas 1 relevante e ele aparece no top-10, Precision@10 = 1/10 = 0.1,
    mesmo que o Recall seja perfeito (1.0). Isso é comportamento esperado e matematicamente correto
    para datasets com relevância esparsa (poucos relevantes por query).

Observações:
- Em `nDCG@k`, utilizamos ganhos (coluna `score` de `qrels`) permitindo graus de relevância (ex.: 0/1/2). Para outras métricas, tratamos relevância binária (`score > 0`).
- Se `qrels` para uma `query_id` estiver vazio, as métricas daquela consulta tendem a 0.

## O que o avaliador faz

`evaluate_predictions(preds, qrels, ks=(1,3,5,10))`:
- Constrói um mapa de relevância por consulta a partir de `qrels`.
- Para cada `query_id` presente em `preds`, computa MRR, nDCG, MAP, Recall e Precision em cada `k` informado.
- Retorna um DataFrame agregado (média por `k`) com colunas: `k, MRR, nDCG, MAP, Recall, Precision`.