# Automatos Celulares

Repositório de trabalhos de autômatos celulares, com destaque para o projeto `forest_fire_CA` em `implementacao_final/forest_fire_CA`.

## Projeto `forest_fire_CA`

Simulação e classificação de propagação de incêndios florestais com:
- geração de dataset sintético a partir do autômato celular;
- treino de um modelo Transformer em PyTorch;
- avaliação com matriz de confusão e relatório de classificação;
- visualizações de distribuição, curva de perda e animação GIF.

## Estrutura principal

- `implementacao_final/forest_fire_CA/main.py` — executa o fluxo completo;
- `implementacao_final/forest_fire_CA/train.py` — treina e avalia o modelo;
- `implementacao_final/forest_fire_CA/data/` — geração e carga dos dados;
- `implementacao_final/forest_fire_CA/simulation/` — regras do autômato;
- `implementacao_final/forest_fire_CA/models/` — modelo Transformer;
- `implementacao_final/forest_fire_CA/visualization/` — gráficos e animações;
- `implementacao_final/forest_fire_CA/results/` — artefatos gerados.

## Requisitos

- Python 3.13+
- Dependências listadas em `implementacao_final/forest_fire_CA/requirements.txt`

## Como executar

Instale as dependências:

```powershell
pip install -r implementacao_final/forest_fire_CA/requirements.txt
```

Execute o fluxo completo:

```powershell
python implementacao_final/forest_fire_CA/main.py
```

Ou rode apenas o treino:

```powershell
python implementacao_final/forest_fire_CA/train.py
```

## Saídas geradas

Os principais arquivos são salvos em `implementacao_final/forest_fire_CA/results/`:
- `state_distribution.png`
- `loss_curve.png`
- `confusion_matrix.png`
- `fire_spread.gif`
- `classification_report.txt`

## Observações

- O projeto usa caminhos absolutos internos ao pacote para evitar problemas com `cwd`.
- O dataset e os resultados podem ser regenerados a qualquer momento executando `main.py`.
