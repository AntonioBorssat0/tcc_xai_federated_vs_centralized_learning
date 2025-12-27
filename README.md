# Eles aprendem diferente? Investigando padrões de aprendizado em modelos federados globais e centralizados com técnicas de Explainable AI (XAI)

Esse projeto tem o objetivo de responder à seguinte pergunta: modelos federados e centralizados aprendem os mesmos padrões de decisão? Para isso, foi criado um cenário controlado onde modelos MLP e XGBoost foram treinados de forma centralizada e federada, e depois passaram pelo mesmo processo de avaliação de explicabilidade utilizando as técnicas SHAP e LIME. Os resultados mostraram que os modelos FL mantiveram performance competitiva aos centralizados e aprenderam padrões de decisão semelhantes.

Nesse arquivo você encontrará instruções detalhadas para reproduzir todos os experimentos, desde a preparação dos dados, treinamento dos modelos, até as análises de explicabilidade e visualizações.

O artigo do projeto pode ser encontrado abaixo:

[![Ler Artigo](https://img.shields.io/badge/PDF-Ler_Artigo_Completo-EC1C24?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](https://github.com/AntonioBorssat0/tcc_xai_federated_vs_centralized_learning/blob/c17684a212728c17757d0ccd687323551191f415/artigo_do_projeto/Eles%20aprendem%20diferente%20-%20Investigando%20padr%C3%B5es%20de%20aprendizado%20em%20modelos%20federados%20globais%20e%20centralizados%20com%20t%C3%A9cnicas%20de%20Explainable%20AI%20(XAI).pdf)

## Sumário

1. [Visão Geral](#visão-geral)
2. [Estrutura do Projeto](#estrutura-do-projeto)
3. [Requisitos e Instalação](#requisitos-e-instalação)
4. [Preparação dos Dados](#preparação-dos-dados)
5. [Treinamento dos Modelos](#treinamento-dos-modelos)
6. [Análises de Explicabilidade](#análises-de-explicabilidade)
7. [Notebooks de Visualização](#notebooks-de-visualização)
8. [Arquivos Gerados](#arquivos-gerados)

---

## Visão Geral

### Objetivo

Comparar o desempenho e a interpretabilidade de modelos de classificação binária (previsão de chuva) treinados de forma centralizada versus federada, utilizando:

- **Modelos Centralizados**: MLP (Multi-Layer Perceptron) e XGBoost treinados com todos os dados em um único local
- **Modelos Federados**: MLP com FedAvg e XGBoost com estratégias Bagging e Cyclic, onde cada localização geográfica representa um cliente federado

### Dataset

O projeto utiliza o dataset Rain in Australia (Kaggle), contendo registros meteorológicos de 44 estações australianas. A variável alvo é `RainTomorrow`, indicando se haverá chuva no dia seguinte.

### Modelos Implementados

| Modelo | Tipo | Descrição |
|--------|------|-----------|
| MLP Centralizado | Rede Neural | Treinamento tradicional com todos os dados |
| XGBoost Centralizado | Gradient Boosting | Treinamento tradicional com todos os dados |
| MLP Federado (FedAvg) | Rede Neural | Agregação por média ponderada dos gradientes |
| XGBoost Federado (Bagging) | Gradient Boosting | Clientes treinam em paralelo, árvores agregadas |
| XGBoost Federado (Cyclic) | Gradient Boosting | Clientes treinam sequencialmente |

---

## Estrutura do Projeto

```
projeto_tcc/
├── centralized_training/       # Scripts de treinamento centralizado
│   ├── train_centralized_mlp.py
│   ├── train_centralized_xgboost.py
│   └── models/                 # Modelos salvos
├── datasets/                   # Dados e scripts de preparação
│   ├── rain_australia/         # Dataset original e limpo
│   │   ├── weatherAUS.csv
│   │   ├── weatherAUS_cleaned.csv
│   │   └── by_location/        # Dados particionados por localização
│   ├── create_train_test_split.py
│   ├── create_val_split.py
│   └── validate_split.py
├── flwr-mlp/                   # Projeto Flower para MLP federado
│   ├── federated_mlp/
│   └── models/
├── flwr-xgboost/               # Projeto Flower para XGBoost federado
│   ├── federated_xgboost/
│   └── models/
├── xAI/                        # Scripts de explicabilidade
│   ├── shap_analysis_*.py      # Análises SHAP (5 scripts)
│   ├── lime_analysis_*.py      # Análises LIME (5 scripts)
│   ├── shap_results_*/         # Resultados SHAP
│   └── lime_results_*/         # Resultados LIME
├── notebooks/                  # Jupyter notebooks para visualização
├── utils/                      # Funções utilitárias compartilhadas
└── pyproject.toml              # Configuração Poetry principal
```

---

## Requisitos e Instalação

### Pré-requisitos

- Python 3.11.7
- Poetry (gerenciador de dependências)
- Git

### Instalação do Poetry

```bash
# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Linux/macOS
curl -sSL https://install.python-poetry.org | python3 -
```

Adicione o Poetry ao PATH conforme instruções exibidas após instalação.

### Configuração do Ambiente

```bash
# Clone o repositório (se aplicável)
cd projeto_tcc

# Instalar dependências do projeto principal
poetry install --no-root

# Instalar dependências do projeto MLP federado
cd flwr-mlp
poetry run pip install -e .
cd ..

# Instalar dependências do projeto XGBoost federado
cd flwr-xgboost
poetry run pip install -e .
cd ..
```

### Verificar Instalação

```bash
# Verificar se Flower está instalado
poetry run flwr --version

# Verificar dependências principais
poetry run python -c "import torch; import xgboost; import shap; import lime; print('OK')"
```

### Dependências Principais

O projeto utiliza as seguintes bibliotecas:

- `flwr[simulation]` >= 1.22.0: Framework de aprendizado federado
- `torch` >= 2.8.0: Redes neurais (MLP)
- `xgboost` == 3.0.0: Gradient boosting
- `shap` == 0.49.1: Explicabilidade SHAP
- `lime` >= 0.2.0: Explicabilidade LIME
- `optuna` >= 4.5.0: Otimização de hiperparâmetros
- `scikit-learn` == 1.7.2: Métricas e preprocessamento
- `pandas`, `matplotlib`, `seaborn`: Manipulação e visualização

---

## Preparação dos Dados

### 1. Limpeza dos Dados (Opcional)

Se precisar reprocessar o dataset original:

```bash
cd notebooks
# Abrir e executar data_cleaning.ipynb no Jupyter
jupyter notebook data_cleaning.ipynb
```

O notebook gera `weatherAUS_cleaned.csv` a partir de `weatherAUS.csv`, tratando valores ausentes e codificando variáveis categóricas.

### 2. Criar Splits de Treino/Teste

```bash
cd datasets

# Gera train_indices.csv e test_indices.csv
poetry run python create_train_test_split.py
```

Este script:
- Divide os dados em 80% treino e 20% teste
- Usa estratificação por classe (RainTomorrow)
- Salva os índices para reprodutibilidade

### 3. Criar Split de Validação

```bash
# Gera val_indices.csv a partir do treino
poetry run python create_val_split.py
```

Separa 10% do conjunto de treino para validação durante otimização de hiperparâmetros.

### 4. Validar Splits (Opcional)

```bash
poetry run python validate_split.py
```

Verifica se não há vazamento de dados entre conjuntos e exibe estatísticas.

### 5. Particionamento por Localização

Os dados em `rain_australia/by_location/` contêm um arquivo CSV por estação meteorológica. Estes são usados automaticamente pelo treinamento federado, onde cada localização representa um cliente.

---

## Treinamento dos Modelos

### Ordem de Execução

Os modelos devem ser treinados na seguinte ordem devido às dependências de hiperparâmetros:

1. MLP Centralizado
2. XGBoost Centralizado
3. Sincronizar hiperparâmetros
4. MLP Federado
5. XGBoost Federado (Bagging)
6. XGBoost Federado (Cyclic)

### 1. Treinar MLP Centralizado

```bash
cd centralized_training
poetry run python train_centralized_mlp.py
```

- **Saídas**:
  - `models/mlp/centralized_model_best.joblib`
  - `models/mlp/best_hyperparameters.json`
  - `models/mlp/training_history.csv`

O script utiliza Optuna para buscar os melhores hiperparâmetros (learning rate, hidden layers, dropout, etc.) e treina o modelo final com a melhor configuração.

### 2. Treinar XGBoost Centralizado

```bash
poetry run python train_centralized_xgboost.py
```

- **Saídas**:
  - `models/xgboost/centralized_model_best.joblib`
  - `models/xgboost/best_hyperparameters.json`
  - `models/xgboost/training_history.csv`

### 3. Sincronizar Hiperparâmetros

Este passo copia os melhores hiperparâmetros encontrados no treinamento centralizado para os projetos federados, garantindo comparação justa:

```bash
cd ..

# Windows PowerShell
.\sync_hyperparameters.ps1

# Alternativa manual: editar os arquivos pyproject.toml nas pastas flwr-mlp e flwr-xgboost
```

Se preferir fazer manualmente, copie os valores de `best_hyperparameters.json` para a seção `[tool.flwr.app.config]` dos respectivos arquivos:
- `flwr-mlp/pyproject.toml`
- `flwr-xgboost/pyproject.toml`

### 4. Treinar MLP Federado

```bash
cd flwr-mlp
poetry run flwr run .
```

- **Estratégia**: FedAvg (Federated Averaging)
- **Saídas**:
  - `models/global_model_final.joblib`
  - `models/global_model_final.pt`
  - `models/training_history.csv`
  - `models/checkpoints/round_*.pt`

### 5. Treinar XGBoost Federado (Bagging)

```bash
cd ../flwr-xgboost
poetry run flwr run . --run-config "strategy='bagging'"
```

- **Estratégia**: Todos os clientes treinam em paralelo, árvores são agregadas
- **Saídas**:
  - `models/global_model_bagging_final.joblib`
  - `models/global_model_bagging_final.json`
  - `models/training_history_bagging.csv`

### 6. Treinar XGBoost Federado (Cyclic)

```bash
poetry run flwr run . --run-config "strategy='cyclic'"
```

- **Estratégia**: Clientes treinam um por vez, modelo é passado sequencialmente
- **Saídas**:
  - `models/global_model_cyclic_final.joblib`
  - `models/global_model_cyclic_final.json`
  - `models/training_history_cyclic.csv`

### 7. Gerar Métricas Consolidadas

Após treinar todos os modelos:

```bash
cd ../utils
poetry run python generate_test_metrics.py
```

- **Saída**: `test_metrics_all_models.csv` na raiz do projeto

Este script avalia todos os 5 modelos no mesmo conjunto de teste, calculando métricas comparáveis (Accuracy, Precision, Recall, F1, ROC-AUC, AUCPR, MCC).

---

## Análises de Explicabilidade

### Análises SHAP

SHAP (SHapley Additive exPlanations) calcula a contribuição de cada feature para as predições individuais.

```bash
cd xAI

# MLP Centralizado
poetry run python shap_analysis_centralized_mlp.py

# XGBoost Centralizado
poetry run python shap_analysis_centralized_xgboost.py

# MLP Federado
poetry run python shap_analysis_federated_mlp.py

# XGBoost Federado (Bagging)
poetry run python shap_analysis_federated_xgboost_bagging.py

# XGBoost Federado (Cyclic)
poetry run python shap_analysis_federated_xgboost_cyclic.py
```

- **Método MLP**: GradientExplainer
- **Método XGBoost**: TreeExplainer

**Saídas por modelo** (em `shap_results_centralized/` ou `shap_results_federated/`):
- `feature_importance_all.csv`: Importância média por feature
- `feature_importance_class_0.csv`: Importância para classe "Não chove"
- `feature_importance_class_1.csv`: Importância para classe "Chove"
- `comparison_class_0_vs_1.csv`: Comparação entre classes
- `shap_values.npy`: Valores SHAP brutos
- `feature_values.npy`: Valores das features
- `feature_names.json`: Nomes das features

### Gráfico Comparativo SHAP

```bash
poetry run python generate_shap_global_comparison.py
```

Gera visualizações comparando a importância de features entre todos os modelos.

### Análises LIME

LIME (Local Interpretable Model-agnostic Explanations) gera explicações locais por amostra.

```bash
# MLP Centralizado
poetry run python lime_analysis_centralized_mlp.py

# XGBoost Centralizado
poetry run python lime_analysis_centralized_xgboost.py

# MLP Federado
poetry run python lime_analysis_federated_mlp.py

# XGBoost Federado (Bagging)
poetry run python lime_analysis_federated_xgboost_bagging.py

# XGBoost Federado (Cyclic)
poetry run python lime_analysis_federated_xgboost_cyclic.py
```

**Saídas por modelo** (em `lime_results_centralized/` ou `lime_results_federated/`):
- `feature_importance_all.csv`: Importância agregada
- `feature_importance_class_*.csv`: Por classe
- `lime_instance_weights.csv`: Pesos por instância
- `lime_instance_abs_weights.csv`: Pesos absolutos
- `lime_instance_fidelity.csv`: Métricas de fidelidade
- `comparison_class_0_vs_1.csv`: Comparação entre classes
- `summary.md`: Resumo textual

---

## Notebooks de Visualização

Os notebooks Jupyter fornecem análises visuais e estatísticas dos resultados.

### 1. Limpeza de Dados

```
notebooks/data_cleaning.ipynb
```

Processa o dataset original, trata valores ausentes e gera `weatherAUS_cleaned.csv`.

### 2. Comparação de Modelos

```
notebooks/model_evaluation_comparison.ipynb
```

Compara métricas de performance dos 5 modelos:
- Tabelas comparativas
- Gráficos de barras por métrica
- Matrizes de confusão
- Curvas ROC e Precision-Recall

### 3. Resultados e Discussão

```
notebooks/resultados_e_discussao_completo.ipynb
```

Análise aprofundada incluindo:
- Comparação SHAP entre modelos
- Análise de correlação de Spearman entre rankings de features
- Visualizações

### Executar Notebooks

```bash
cd notebooks
poetry run jupyter notebook
```

Ou abrir diretamente no VS Code com a extensão Jupyter.

---

## Arquivos Gerados

### Modelos Treinados

| Arquivo | Descrição |
|---------|-----------|
| `centralized_training/models/mlp/centralized_model_best.joblib` | MLP centralizado final |
| `centralized_training/models/xgboost/centralized_model_best.joblib` | XGBoost centralizado final |
| `flwr-mlp/models/global_model_final.joblib` | MLP federado final |
| `flwr-xgboost/models/global_model_bagging_final.joblib` | XGBoost federado (Bagging) |
| `flwr-xgboost/models/global_model_cyclic_final.joblib` | XGBoost federado (Cyclic) |

### Hiperparâmetros

| Arquivo | Descrição |
|---------|-----------|
| `centralized_training/models/mlp/best_hyperparameters.json` | Hiperparâmetros MLP |
| `centralized_training/models/xgboost/best_hyperparameters.json` | Hiperparâmetros XGBoost |

### Métricas e Resultados

| Arquivo | Descrição |
|---------|-----------|
| `test_metrics_all_models.csv` | Métricas consolidadas dos 5 modelos |
| `notebooks/model_comparison_results.csv` | Resultados da comparação |
| `notebooks/spearman_correlation_results.csv` | Correlação entre rankings SHAP |

### Resultados SHAP

Diretórios `xAI/shap_results_centralized/` e `xAI/shap_results_federated/` contendo:
- CSVs de importância de features por modelo
- Arrays NumPy com valores SHAP brutos
- Metadados em JSON

### Resultados LIME

Diretórios `xAI/lime_results_centralized/` e `xAI/lime_results_federated/` contendo:
- CSVs de importância agregada
- Pesos por instância
- Métricas de fidelidade
- Resumos em Markdown

### Histórico de Treinamento

| Arquivo | Descrição |
|---------|-----------|
| `centralized_training/models/*/training_history.csv` | Métricas por época/trial |
| `flwr-mlp/models/training_history.csv` | Métricas por round federado |
| `flwr-xgboost/models/training_history_*.csv` | Métricas por estratégia |

---

## Notas Adicionais

### Reprodutibilidade

Todos os scripts utilizam seeds fixas (RANDOM_STATE = 42) para garantir reprodutibilidade. Os arquivos de índices em `datasets/` preservam os splits exatos utilizados.

### Troubleshooting

**Erro de importação de módulos**:
```bash
# Garantir que está no diretório correto
cd projeto_tcc
poetry install --no-root
```

**Erro no XGBoost base_score**:
Os scripts incluem correção automática para o bug de serialização do base_score em formato JSON.

**Memória insuficiente em SHAP**:
Reduzir o número de amostras nos scripts de análise (padrão: 2000).

---

Autor: Antonio Borssato
