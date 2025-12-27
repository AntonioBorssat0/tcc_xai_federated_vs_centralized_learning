"""Utilitários de Treinamento para o Projeto.
Funções compartilhadas para treinamento centralizado e federado.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_recall_fscore_support,
    matthews_corrcoef,
    confusion_matrix,
    average_precision_score
)

RANDOM_STATE = 42


def print_section(title: str, width: int = 80):
    """Imprime um cabeçalho de seção formatado."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def load_train_test_data(
    data_path: Path,
    train_indices_path: Path,
    test_indices_path: Path
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Carrega dataset e índices de treino/teste (padronizado).
    
    Returns:
        df: DataFrame completo
        train_indices: Array de índices de treino
        test_indices: Array de índices de teste
    """
    print("\nCarregando dados...")
    df = pd.read_csv(data_path)
    print(f"   Total de amostras: {len(df):,}")
    
    print("\nCarregando índices de treino/teste...")
    train_indices_df = pd.read_csv(train_indices_path)
    test_indices_df = pd.read_csv(test_indices_path)
    
    train_indices = train_indices_df['index'].values
    test_indices = test_indices_df['index'].values
    
    print(f"   Treino: {len(train_indices):,} amostras ({len(train_indices)/len(df)*100:.1f}%)")
    print(f"   Teste: {len(test_indices):,} amostras ({len(test_indices)/len(df)*100:.1f}%)")
    
    return df, train_indices, test_indices


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calcula métricas de avaliação padrão.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Labels preditos (binário)
        y_probs: Probabilidades preditas (para AUC)
    
    Returns:
        Dicionário com métricas
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    # Matthews Correlation Coefficient (melhor para dados desbalanceados)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'mcc': float(mcc),
        'confusion_matrix': cm.tolist(),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }
    
    if y_probs is not None:
        auc = roc_auc_score(y_true, y_probs)
        avg_precision = average_precision_score(y_true, y_probs)
        metrics['auc'] = float(auc)
        metrics['avg_precision'] = float(avg_precision)
    
    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Imprime métricas em formato padronizado."""
    print(f"\n{prefix}Métricas de avaliação:")
    if 'accuracy' in metrics:
        print(f"   Accuracy     : {metrics['accuracy']:.4f}")
    if 'auc' in metrics:
        print(f"   AUC          : {metrics['auc']:.4f}")
    if 'avg_precision' in metrics:
        print(f"   Avg Precision: {metrics['avg_precision']:.4f}")
    if 'precision' in metrics:
        print(f"   Precision    : {metrics['precision']:.4f}")
    if 'recall' in metrics:
        print(f"   Recall       : {metrics['recall']:.4f}")
    if 'f1' in metrics:
        print(f"   F1-Score     : {metrics['f1']:.4f}")
    if 'mcc' in metrics:
        print(f"   MCC          : {metrics['mcc']:.4f}")
    if 'tn' in metrics and 'fp' in metrics and 'fn' in metrics and 'tp' in metrics:
        print("\nConfusion Matrix:")
        print(f"TN: {metrics['tn']}  FP: {metrics['fp']}")
        print(f"FN: {metrics['fn']}  TP: {metrics['tp']}")


def save_model_joblib(
    save_dir: Path,
    model_name: str,
    model_data: Dict[str, Any],
    also_save_json: bool = True
) -> None:
    """
    Salva modelo em formato joblib padronizado.
    
    Args:
        save_dir: Diretório para salvar modelo
        model_name: Nome base para o modelo (sem extensão)
        model_data: Dicionário contendo modelo e metadados
        also_save_json: Se deve também salvar hiperparâmetros como JSON
    
    Estrutura esperada de model_data:
        {
            'best_model': objeto do modelo (modelo PyTorch, booster XGBoost, etc.),
            'best_params': dict de hiperparâmetros,
            'best_score': float,
            'study': optuna.Study (opcional),
            'scaler': scaler sklearn (opcional),
            'feature_names': lista de nomes das features,
            'train_auc': float,
            'test_auc': float,
            'metadata': {
                'train_date': str,
                'model_type': str,
                'framework': str,
                'accuracy': float,
                'auc': float,
                'precision': float,
                'recall': float,
                'f1': float,
                ...
            }
        }
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nSalvando modelo...")
    
    # Salvar joblib
    joblib_path = save_dir / f"{model_name}.joblib"
    joblib.dump(model_data, joblib_path)
    print(f"   {joblib_path}")
    
    # Salvar JSON (apenas hiperparâmetros)
    if also_save_json and 'best_params' in model_data:
        json_path = save_dir / "best_hyperparameters.json"
        
        json_data = {
            **model_data['best_params'],
            'metrics': {
                k: v for k, v in model_data['metadata'].items()
                if k in ['accuracy', 'auc', 'precision', 'recall', 'f1', 'loss']
            }
        }
        
        # Adicionar outros campos relevantes
        if 'feature_names' in model_data:
            json_data['num_features'] = len(model_data['feature_names'])
        
        if 'input_size' in model_data.get('metadata', {}):
            json_data['input_size'] = model_data['metadata']['input_size']
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"   {json_path}")
    
    print(f"   Modelo salvo com sucesso!")


def print_class_distribution(y: np.ndarray, label: str = "Dataset"):
    """Imprime distribuição de classes em formato padronizado."""
    class_0 = np.sum(y == 0)
    class_1 = np.sum(y == 1)
    total = len(y)
    
    print(f"\nDistribuição de classes ({label}):")
    print(f"   Classe 0 (No Rain): {class_0:,} ({class_0/total*100:.1f}%)")
    print(f"   Classe 1 (Rain)   : {class_1:,} ({class_1/total*100:.1f}%)")
