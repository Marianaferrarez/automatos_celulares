# Automatos Celulares

Repositório de trabalhos de autômatos celulares, com destaque para o projeto `forest_fire_CA` em `implementacao_final/forest_fire_CA`.

## Projeto `forest_fire_CA`

Simulação e classificação de propagação de incêndios florestais com:
- geração de dataset sintético realista a partir do autômato celular;
- treino de um modelo Transformer em PyTorch;
- avaliação com matriz de confusão e relatório de classificação;
- experimento de limiar (threshold) para análise de sensibilidade;
- comparação espacial entre Transformer-CA e CA analítico;
- visualização interativa com Pygame;
- visualizações de distribuição, curva de perda e animação GIF.

## Estrutura principal

- `implementacao_final/forest_fire_CA/main.py` — executa o fluxo completo;
- `implementacao_final/forest_fire_CA/train.py` — treina e avalia o modelo;
- `implementacao_final/forest_fire_CA/threshold_experiment.py` — experimento de sensibilidade ao limiar;
- `implementacao_final/forest_fire_CA/visualize_pygame.py` — visualização interativa com Pygame;
- `implementacao_final/forest_fire_CA/data/` — geração e carga dos dados;
- `implementacao_final/forest_fire_CA/data/neighborhood.py` — extração de vizinhança para features;
- `implementacao_final/forest_fire_CA/simulation/` — regras do autômato;
- `implementacao_final/forest_fire_CA/simulation/metrics.py` — métricas de avaliação do CA;
- `implementacao_final/forest_fire_CA/models/` — modelo Transformer;
- `implementacao_final/forest_fire_CA/visualization/` — gráficos e animações;
- `implementacao_final/forest_fire_CA/visualization/grid_comparison.py` — comparação Transformer-CA vs CA analítico;
- `implementacao_final/forest_fire_CA/results/` — artefatos gerados.

## Requisitos

- Python 3.13+
- Dependências: numpy, matplotlib, pandas, scikit-learn, torch, pygame
- Listadas em `implementacao_final/forest_fire_CA/requirements.txt`

## Como executar

Instale as dependências:

```powershell
pip install -r implementacao_final/forest_fire_CA/requirements.txt
```

Execute o fluxo completo:

```powershell
python implementacao_final/forest_fire_CA/main.py
```

Rode apenas o treino:

```powershell
python implementacao_final/forest_fire_CA/train.py
```

Rode o experimento de limiar:

```powershell
python implementacao_final/forest_fire_CA/threshold_experiment.py
```

Visualização interativa com Pygame:

```powershell
python implementacao_final/forest_fire_CA/visualize_pygame.py
```

## Saídas geradas

Os principais arquivos são salvos em `implementacao_final/forest_fire_CA/results/`:
- `state_distribution.png`
- `loss_curve.png`
- `confusion_matrix.png`
- `classification_report.txt`
- `fire_spread.gif` — animação da simulação Transformer-CA
- `grid_comparison.png` — comparação espacial entre Transformer-CA e
  a versão analítica do CA rodadas no mesmo ambiente sintético
  (referência, já que não há fogo real para comparar)
- `threshold_experiment.csv` — resultados do experimento de limiar
- `threshold_experiment.png` — gráfico do experimento de limiar

## Observações

- O projeto usa caminhos absolutos internos ao pacote para evitar problemas com `cwd`.
- O dataset e os resultados podem ser regenerados a qualquer momento executando `main.py`.
- O modelo treinado e o scaler são salvos em `implementacao_final/forest_fire_CA/saved_models/`.

## Reproduzir o treino localmente

Use um ambiente virtual e instale as dependências listadas em `implementacao_final/forest_fire_CA/requirements.txt`.

Exemplo (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r implementacao_final/forest_fire_CA/requirements.txt
python implementacao_final/forest_fire_CA/train.py
```

Ou execute o fluxo completo (gera dataset e treina):

```powershell
python implementacao_final/forest_fire_CA/main.py
```

Para a visualização interativa com Pygame:

```powershell
python implementacao_final/forest_fire_CA/visualize_pygame.py
```

Os artefatos gerados irão para `implementacao_final/forest_fire_CA/results/`.
