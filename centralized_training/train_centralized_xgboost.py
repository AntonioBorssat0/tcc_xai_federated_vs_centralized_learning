"""
Treinamento Centralizado XGBoost com Otimização de Hiperparâmetros usando Optuna.
Utiliza a mesma divisão de dados do treinamento federado para garantir comparação justa.
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.impute import SimpleImputer
import optuna
from optuna.trial import TrialState
import time

# Adiciona caminhos necessários para importações
sys.path.append(str(Path(__file__).parent.parent / "flwr-xgboost"))
sys.path.append(str(Path(__file__).parent.parent))

from federated_xgboost.task import _encode_wind_directions_cyclic, LABEL_COL
from utils import (
    load_train_test_data,
    save_model_joblib,
    compute_metrics,
    print_section,
    print_metrics,
    print_class_distribution
)

# Configuração de seed para reprodutibilidade
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Definição de caminhos
script_dir = Path(__file__).parent
project_root = script_dir.parent
data_path = project_root / "datasets" / "rain_australia" / "weatherAUS_cleaned.csv"
train_indices_path = project_root / "datasets" / "train_indices.csv"
val_indices_path = project_root / "datasets" / "val_indices.csv"
test_indices_path = project_root / "datasets" / "test_indices.csv"
models_dir = script_dir / "models" / "xgboost"
models_dir.mkdir(parents=True, exist_ok=True)

print_section("TREINAMENTO CENTRALIZADO XGBOOST COM OPTUNA")
print(f"\nDataset: {data_path.name}")

print(f"\nCarregando índices...")
train_indices = pd.read_csv(train_indices_path)['index'].values
val_indices = pd.read_csv(val_indices_path)['index'].values
test_indices = pd.read_csv(test_indices_path)['index'].values
df = pd.read_csv(data_path)

print(f"   Total amostras: {len(df):,}")
print(f"   Train: {len(train_indices):,}")
print(f"   Val: {len(val_indices):,}")
print(f"   Test: {len(test_indices):,}")

print(f"\nPreprocessando dados...")
df = df.drop(columns=['Location'], errors='ignore')
df = _encode_wind_directions_cyclic(df)

X = df.drop(columns=['RainTomorrow']).values
y = df['RainTomorrow'].values
feature_names = df.drop(columns=['RainTomorrow']).columns.tolist()

print(f"   Features: {len(feature_names)}")
print(f"   Encoding cíclico de vento aplicado (sin/cos)")

# Divisão dos dados usando os índices pré-definidos
X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]
y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]

# Tratamento de valores ausentes com mediana (ajustado apenas nos dados de treino)
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)

print(f"   Valores ausentes tratados com mediana")

# Criação de DMatrix (formato nativo do XGBoost)
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

print_class_distribution(y_train, "Treino")
print_class_distribution(y_val, "Validação")
print_class_distribution(y_test, "Teste")

# Cálculo de scale_pos_weight para balanceamento de classes desbalanceadas
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nBalanceamento de Classes:")
print(f"   Classe 0 (Sem Chuva): {(y_train == 0).sum():,} amostras")
print(f"   Classe 1 (Chuva): {(y_train == 1).sum():,} amostras")
print(f"   Proporção: {scale_pos_weight:.2f}:1")
print(f"   scale_pos_weight aplicado: {scale_pos_weight:.2f}")


def train_and_evaluate(trial, verbose=False):
    """
    Treina o modelo XGBoost com os hiperparâmetros fornecidos e retorna as métricas de teste.
    
    Args:
        trial: Trial do Optuna contendo os hiperparâmetros a serem testados
        verbose: Se True, imprime informações detalhadas durante o treinamento
    
    Returns:
        tuple: (métricas, modelo, histórico de treinamento)
    """
    # Hyperparameters to optimize
    max_depth = trial.suggest_int("max_depth", 4, 6)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    subsample = trial.suggest_float("subsample", 0.6, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 7)
    gamma = trial.suggest_float("gamma", 0.0, 0.5)
    reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True)
    reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True)
    
    # Parâmetros do XGBoost com balanceamento de classes
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['aucpr', 'auc'],
        'scale_pos_weight': scale_pos_weight,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'min_child_weight': min_child_weight,
        'gamma': gamma,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'seed': RANDOM_STATE,
        'verbosity': 0
    }
    
    # Treinamento com early stopping baseado no conjunto de validação
    evals = [(dtrain, 'train'), (dval, 'val')]
    evals_result = {}
    
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=20,
        evals_result=evals_result,
        verbose_eval=False
    )
    
    # Predição no conjunto de teste para avaliação final
    y_pred_probs = bst.predict(dtest)
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    # Cálculo das métricas no conjunto de teste
    metrics = compute_metrics(y_test, y_pred, y_pred_probs)
    
    if verbose:
        total_rounds = len(evals_result['train']['aucpr'])
        stopped_early = total_rounds < 500
        
        print(f"      Melhor iteração: {bst.best_iteration} / {total_rounds} rounds")
        if stopped_early:
            print(f"      Early stopping ativado (parou em {total_rounds}/500)")
        else:
            print(f"      Executou todas as {total_rounds} rounds")
        
        print(f"      Train AUCPR: {evals_result['train']['aucpr'][bst.best_iteration]:.4f}")
        print(f"      Val AUCPR: {evals_result['val']['aucpr'][bst.best_iteration]:.4f}")
        print(f"      Train AUC: {evals_result['train']['auc'][bst.best_iteration]:.4f}")
        print(f"      Val AUC: {evals_result['val']['auc'][bst.best_iteration]:.4f}")
        
        # Indicador de overfitting
        train_val_gap = evals_result['train']['aucpr'][bst.best_iteration] - evals_result['val']['aucpr'][bst.best_iteration]
        if train_val_gap > 0.1:
            print(f"      Overfitting detectado (diferença train-val: {train_val_gap:.4f})")
        elif train_val_gap > 0.05:
            print(f"      Overfitting leve (diferença train-val: {train_val_gap:.4f})")
        else:
            print(f"      Boa generalização (diferença train-val: {train_val_gap:.4f})")
    
    # Preparação do histórico de treinamento
    history = {
        'iteration': list(range(1, len(evals_result['train']['aucpr']) + 1)),
        'train_aucpr': evals_result['train']['aucpr'],
        'val_aucpr': evals_result['val']['aucpr'],
        'train_auc': evals_result['train']['auc'],
        'val_auc': evals_result['val']['auc']
    }
    
    return metrics, bst, history


# Configuração da otimização de hiperparâmetros
N_TRIALS = 200
TIMEOUT = 10800  # 3 horas

print(f"\nIniciando otimização de hiperparâmetros com Optuna...")
print(f"   Trials: {N_TRIALS}")
print(f"   Timeout: {TIMEOUT//60} minutos")
print(f"   Objetivo: Maximizar AUCPR (Average Precision)")
print(f"   AUCPR é mais apropriado para datasets desbalanceados")
print()

def objective(trial):
    """Função objetivo do Optuna que retorna a métrica a ser otimizada."""
    metrics, _, _ = train_and_evaluate(trial, verbose=False)
    return metrics['avg_precision']

# Criação do estudo Optuna
study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
)

# Execução da otimização
start_time = time.time()
study.optimize(
    objective,
    n_trials=N_TRIALS,
    timeout=TIMEOUT,
    show_progress_bar=True,
    n_jobs=1
)
optim_time = time.time() - start_time

print(f"\nOtimização concluída em {optim_time/60:.1f} minutos!")

# Resultados da otimização
best_trial = study.best_trial
print(f"\nMelhores hiperparâmetros encontrados:")
print(f"   AUCPR (Average Precision): {best_trial.value:.4f}")
print(f"   Hiperparâmetros:")
for key, value in best_trial.params.items():
    print(f"      {key}: {value}")

print(f"\nTreinando modelo final com melhores hiperparâmetros...")

# Criação de um trial com os melhores parâmetros para treinamento final
class BestParamsTrial:
    def __init__(self, params):
        self.params = params
    
    def suggest_int(self, name, *args, **kwargs):
        return self.params[name]
    
    def suggest_float(self, name, *args, **kwargs):
        return self.params[name]

best_params_trial = BestParamsTrial(best_trial.params)
final_metrics, final_model, final_history = train_and_evaluate(best_params_trial, verbose=True)

print(f"\nResumo do Early Stopping:")
print(f"   Melhor iteração: {final_model.best_iteration}")
print(f"   Total de iterações: {len(final_history['iteration'])}")
print(f"   Rounds economizadas com early stopping: {500 - len(final_history['iteration'])}")
print(f"   Eficiência: {(1 - len(final_history['iteration'])/500)*100:.1f}% de redução no tempo de treinamento")

# Impressão das métricas finais
print_metrics(final_metrics)

# Cálculo das métricas no conjunto de treino para registro completo
y_train_pred_probs = final_model.predict(dtrain)
train_metrics = compute_metrics(y_train, (y_train_pred_probs > 0.5).astype(int), y_train_pred_probs)

# Preparação dos dados do modelo para salvamento
best_params_with_scale = best_trial.params.copy()
best_params_with_scale['scale_pos_weight'] = float(scale_pos_weight)

model_data = {
    "best_model": final_model,
    "best_params": best_params_with_scale,
    "best_score": study.best_value,
    "study": study,
    "feature_names": feature_names,
    "train_auc": train_metrics['auc'],
    "test_auc": final_metrics['auc'],
    "metadata": {
        "train_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "model_type": "xgboost",
        "framework": "xgboost",
        "num_features": len(feature_names),
        "best_iteration": int(final_model.best_iteration),
        "accuracy": final_metrics['accuracy'],
        "auc": final_metrics['auc'],
        "precision": final_metrics['precision'],
        "recall": final_metrics['recall'],
        "f1": final_metrics['f1'],
        "n_trials": len(study.trials),
        "optimization_time_minutes": round(optim_time / 60, 2),
        "scale_pos_weight": float(scale_pos_weight)
    }
}

# Salvamento do modelo usando função utilitária
save_model_joblib(
    save_dir=models_dir,
    model_name="centralized_model_best",
    model_data=model_data,
    also_save_json=True
)

print("\nSalvando histórico de treinamento...")
history_df = pd.DataFrame(final_history)
history_csv_path = models_dir / "training_history.csv"
history_df.to_csv(history_csv_path, index=False)
print(f"   {history_csv_path}")

history_json_path = models_dir / "training_history.json"
history_df.to_json(history_json_path, orient='records', indent=2)
print(f"   {history_json_path}")
print(f"   Histórico salvo ({len(final_history['iteration'])} iterações, melhor: {final_model.best_iteration})")

print_section("TREINAMENTO CONCLUÍDO COM SUCESSO!")

