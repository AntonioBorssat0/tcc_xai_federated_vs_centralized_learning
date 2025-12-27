"""
Treinamento Centralizado MLP com Otimização de Hiperparâmetros usando Optuna.
Utiliza a mesma divisão de dados do treinamento federado para garantir comparação justa.
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.trial import TrialState
import time

# Adiciona caminhos necessários para importações
sys.path.append(str(Path(__file__).parent.parent / "flwr-mlp"))
sys.path.append(str(Path(__file__).parent.parent))

from federated_mlp.task import WeatherMLP, prepare_weather_data, LABEL_COL
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
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Definição de caminhos
script_dir = Path(__file__).parent
project_root = script_dir.parent
data_path = project_root / "datasets" / "rain_australia" / "weatherAUS_cleaned.csv"
train_indices_path = project_root / "datasets" / "train_indices.csv"
val_indices_path = project_root / "datasets" / "val_indices.csv"
test_indices_path = project_root / "datasets" / "test_indices.csv"
models_dir = script_dir / "models" / "mlp"
models_dir.mkdir(exist_ok=True)

# Configuração do device (GPU se disponível, caso contrário CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print_section("TREINAMENTO CENTRALIZADO MLP COM OPTUNA")
print(f"\nDevice: {device}")
print(f"Dataset: {data_path.name}")

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
df_processed = prepare_weather_data(df.copy(), use_location=False)

X = df_processed.drop(columns=[LABEL_COL]).values
y = df_processed[LABEL_COL].values
feature_names = df_processed.drop(columns=[LABEL_COL]).columns.tolist()

print(f"   Features: {len(feature_names)}")

# Divisão dos dados usando os índices pré-definidos
X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]
y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]

# Normalização usando StandardScaler (ajustado apenas nos dados de treino)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print(f"   Normalização aplicada (StandardScaler)")
print(f"   Train mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
print(f"   Val mean: {X_val.mean():.4f}, std: {X_val.std():.4f}")
print(f"   Test mean: {X_test.mean():.4f}, std: {X_test.std():.4f}")

# Conversão para tensores PyTorch
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

print_class_distribution(y_train, "Treino")
print_class_distribution(y_val, "Validação")
print_class_distribution(y_test, "Teste")

# Cálculo de pesos para balanceamento de classes desbalanceadas
class_counts = np.bincount(y_train.astype(int))
class_weights = len(y_train) / (len(class_counts) * class_counts)
pos_weight_value = class_weights[1] / class_weights[0]
print(f"\nBalanceamento de Classes:")
print(f"   Classe 0 (Sem Chuva): {class_counts[0]:,} amostras")
print(f"   Classe 1 (Chuva): {class_counts[1]:,} amostras")
print(f"   Proporção: {class_counts[0]/class_counts[1]:.2f}:1")
print(f"   pos_weight aplicado: {pos_weight_value:.2f}")


def train_and_evaluate(trial, verbose=False):
    """
    Treina o modelo com os hiperparâmetros fornecidos e retorna as métricas de teste.
    
    Args:
        trial: Trial do Optuna contendo os hiperparâmetros a serem testados
        verbose: Se True, imprime informações detalhadas durante o treinamento
    
    Returns:
        tuple: (métricas, modelo, histórico de treinamento)
    """
    # Hyperparameters to optimize
    hidden1 = trial.suggest_int("hidden1", 64, 256, step=32)
    hidden2 = trial.suggest_int("hidden2", 32, 128, step=16)
    hidden3 = trial.suggest_int("hidden3", 16, 64, step=16)
    dropout1 = trial.suggest_float("dropout1", 0.1, 0.5)
    dropout2 = trial.suggest_float("dropout2", 0.1, 0.4)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    
    # Criação do modelo
    input_size = len(feature_names)
    model = WeatherMLP(
        input_size=input_size,
        hidden1=hidden1,
        hidden2=hidden2,
        hidden3=hidden3,
        dropout1=dropout1,
        dropout2=dropout2
    ).to(device)
    
    # Função de perda com peso de classe para dados desbalanceados
    pos_weight = torch.FloatTensor([pos_weight_value]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Criação dos DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Loop de treinamento com early stopping baseado no conjunto de validação
    max_epochs = 150
    patience = 15
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    
    # Histórico de treinamento
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_aucpr': [],
        'val_auc': []
    }
    
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(batch_X)
        
        train_loss /= len(train_dataset)
        
        # Avaliação no conjunto de validação
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * len(batch_X)
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        val_loss /= len(val_dataset)
        val_accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
        
        # Cálculo de métricas adicionais para o histórico
        all_probs_epoch = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                probs = torch.sigmoid(outputs)
                all_probs_epoch.extend(probs.cpu().numpy())
        
        from sklearn.metrics import roc_auc_score, average_precision_score
        val_auc = roc_auc_score(all_labels, all_probs_epoch)
        val_aucpr = average_precision_score(all_labels, all_probs_epoch)
        
        # Armazenamento no histórico
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_aucpr'].append(val_aucpr)
        history['val_auc'].append(val_auc)
        
        # Reporta valor intermediário para pruning (AUCPR é mais apropriado para dados desbalanceados)
        trial.report(val_aucpr, epoch)
        
        # Verifica se o trial deve ser interrompido
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # Early stopping baseado na perda de validação
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            best_model_state = model.state_dict()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"      Epoch {epoch+1:3d}: NOVO MELHOR - Val Loss={val_loss:.4f}, Val AUCPR={val_aucpr:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"      Early stopping na epoch {epoch+1}")
                    print(f"      Melhor epoch: {best_epoch} ({patience} epochs atrás)")
                    print(f"      Melhor val loss: {best_val_loss:.4f}")
                break
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"      Epoch {epoch+1:3d}: Patience {patience_counter}/{patience} - Val Loss={val_loss:.4f}, Val AUCPR={val_aucpr:.4f}")
        
        if verbose and (epoch + 1) % 10 == 0 and patience_counter == 0:
            print(f"      Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val AUCPR={val_aucpr:.4f}, Val AUC={val_auc:.4f}")
    
    # Carrega o melhor modelo e calcula métricas finais
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    # Cálculo das métricas finais no conjunto de teste
    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )
    metrics['loss'] = best_val_loss
    
    return metrics, model, history


# Configuração da otimização de hiperparâmetros
N_TRIALS = 60
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
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
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
    """
    Classe auxiliar de um trial do Optuna com parâmetros fixos.
    Utilizada para treinar o modelo final com os melhores hiperparâmetros encontrados.
    """
    def __init__(self, params):
        self.params = params
    
    def suggest_int(self, name, *args, **kwargs):
        return self.params[name]
    
    def suggest_float(self, name, *args, **kwargs):
        return self.params[name]
    
    def suggest_categorical(self, name, *args, **kwargs):
        return self.params[name]
    
    def report(self, *args, **kwargs):
        pass
    
    def should_prune(self):
        return False

best_params_trial = BestParamsTrial(best_trial.params)
final_metrics, final_model, final_history = train_and_evaluate(best_params_trial, verbose=True)

print(f"\nResumo do Early Stopping:")
total_epochs = len(final_history['epoch'])
best_epoch_idx = final_history['val_aucpr'].index(max(final_history['val_aucpr']))
best_epoch = final_history['epoch'][best_epoch_idx]
print(f"   Melhor epoch: {best_epoch} / {total_epochs}")
print(f"   Melhor val AUCPR: {final_history['val_aucpr'][best_epoch_idx]:.4f}")
if total_epochs < 100:
    print(f"   Early stopping ativado (parou em {total_epochs}/100)")
    print(f"   Epochs economizadas: {100 - total_epochs}")
    print(f"   Eficiência: {(1 - total_epochs/100)*100:.1f}% de redução no tempo de treinamento")
else:
    print(f"   Executou todas as {total_epochs} epochs (considere aumentar max_epochs)")

# Impressão das métricas finais
print_metrics(final_metrics)

# Cálculo das métricas no conjunto de treino para registro completo
final_model.eval()
with torch.no_grad():
    X_train_torch = X_train_tensor.to(device)
    train_logits = final_model(X_train_torch)
    train_probs = torch.sigmoid(train_logits).cpu().numpy()
    train_preds = (train_probs > 0.5).astype(int)
    
train_metrics = compute_metrics(y_train, train_preds, train_probs)

# Preparação dos dados do modelo para salvamento
model_data = {
    "best_model": final_model,
    "best_params": best_trial.params,
    "best_score": study.best_value,
    "study": study,
    "scaler": scaler,
    "feature_names": feature_names,
    "train_auc": train_metrics['auc'],
    "test_auc": final_metrics['auc'],
    "metadata": {
        "train_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "model_type": "mlp_pytorch",
        "framework": "pytorch",
        "input_size": len(feature_names),
        "accuracy": final_metrics['accuracy'],
        "auc": final_metrics['auc'],
        "precision": final_metrics['precision'],
        "recall": final_metrics['recall'],
        "f1": final_metrics['f1'],
        "loss": final_metrics['loss'],
        "n_trials": len(study.trials),
        "optimization_time_minutes": round(optim_time / 60, 2),
        "device": str(device)
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
print(f"   Histórico salvo ({len(final_history['epoch'])} épocas)")

print_section("TREINAMENTO CONCLUÍDO COM SUCESSO!")
