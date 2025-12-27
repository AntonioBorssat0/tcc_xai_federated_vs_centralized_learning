"""Gera métricas consolidadas de teste para todos os modelos.

Carrega todos os 5 modelos treinados e avalia no mesmo conjunto de teste,
gerando um CSV com todas as métricas para comparação justa.

Métricas calculadas:
- Accuracy, Precision, Recall, F1-score
- ROC-AUC, AUCPR (Average Precision)
- Matthews Correlation Coefficient (MCC)
- Elementos da matriz de confusão

Saída: test_metrics_all_models.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
)

# Adicionar paths para imports
sys.path.append(str(Path(__file__).parent.parent / "flwr-mlp"))
sys.path.append(str(Path(__file__).parent.parent / "flwr-xgboost"))
sys.path.append(str(Path(__file__).parent.parent))

from federated_mlp.task import WeatherMLP, prepare_weather_data, LABEL_COL
from federated_xgboost.task import _encode_wind_directions_cyclic
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def load_test_data():
    """Carrega dados de teste (mesmos índices usados no treinamento)."""
    print("\nCarregando dados de teste...")
    project_root = Path(__file__).parent.parent
    
    # Carregar dataset completo
    data_path = project_root / "datasets" / "rain_australia" / "weatherAUS_cleaned.csv"
    df = pd.read_csv(data_path)
    
    # Carregar índices de teste
    test_indices_path = project_root / "datasets" / "test_indices.csv"
    test_indices = pd.read_csv(test_indices_path)['index'].values
    
    df_test = df.iloc[test_indices].copy()
    
    print(f"   Amostras de teste: {len(df_test)}")
    print(f"   Classe positiva: {df_test['RainTomorrow'].sum()} ({df_test['RainTomorrow'].mean()*100:.1f}%)")
    
    return df_test


def prepare_mlp_data(df_test):
    """Prepara dados para modelos MLP (centralizado e federado)."""
    print("\nPreparando dados MLP...")
    
    # Usar mesmo preprocessamento do treinamento
    df_processed = prepare_weather_data(df_test.copy(), use_location=False)
    X_df = df_processed.drop(columns=[LABEL_COL])
    y = df_processed[LABEL_COL].values
    
    # Carregar scaler do treinamento centralizado
    project_root = Path(__file__).parent.parent
    scaler_path = project_root / "centralized_training" / "models" / "mlp" / "centralized_model_best.joblib"
    
    if scaler_path.exists():
        model_data = joblib.load(scaler_path)
        scaler = model_data.get('scaler', None)
        if scaler is not None:
            X = scaler.transform(X_df.values)
            print(f"   Usando scaler do treinamento centralizado")
        else:
            scaler = StandardScaler()
            X = scaler.fit_transform(X_df.values)
            print(f"   [AVISO] Scaler não encontrado, ajustando novo")
    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_df.values)
        print(f"   [AVISO] Modelo não encontrado, ajustando novo scaler")
    
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"   Shape: {X.shape}")
    print(f"   Features: {len(X_df.columns)}")
    
    return X, y, X_df.columns.tolist()


def prepare_xgboost_data(df_test):
    """Prepara dados para modelos XGBoost (centralizado e federado)."""
    print("\nPreparando dados XGBoost...")
    
    df_local = df_test.copy()
    df_local = df_local.drop(columns=['Location'], errors='ignore')
    df_local = _encode_wind_directions_cyclic(df_local)
    
    X_df = df_local.drop(columns=['RainTomorrow'])
    y = df_local['RainTomorrow'].values

    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X_df.values)
    
    print(f"   Shape: {X.shape}")
    print(f"   Features: {len(X_df.columns)}")
    
    return X, y, X_df.columns.tolist()


def load_mlp_model(model_path, device):
    """Carrega modelo MLP de arquivo joblib."""
    print(f"\nCarregando modelo MLP de: {model_path.name}")
    
    if not model_path.exists():
        print(f"   [ERRO] Modelo não encontrado!")
        return None
    
    model_data = joblib.load(model_path)
    model = model_data['best_model']
    model = model.to(device)
    model.eval()
    
    print(f"   Modelo carregado com sucesso")
    return model


def load_xgboost_model(model_path):
    """Carrega modelo XGBoost de arquivo JSON ou JOBLIB."""
    print(f"\nCarregando modelo XGBoost de: {model_path.name}")
    
    if not model_path.exists():
        print(f"   [ERRO] Modelo não encontrado!")
        return None
    
    import xgboost as xgb
    
    # Tentar JOBLIB primeiro (modelos centralizados)
    if model_path.suffix == '.joblib':
        model_data = joblib.load(model_path)
        if isinstance(model_data, dict) and 'best_model' in model_data:
            model = model_data['best_model']
        else:
            model = model_data
    else:
        # Carregar de JSON (modelos federados)
        model = xgb.Booster()
        model.load_model(str(model_path))
    
    print(f"   Modelo carregado com sucesso")
    return model


def evaluate_mlp_model(model, X, y, device):
    """Avalia modelo MLP e retorna predições."""
    print(f"\nAvaliando modelo MLP...")
    
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)
    
    print(f"   Predições geradas")
    return preds, probs


def evaluate_xgboost_model(model, X, y, feature_names=None):
    """Avalia modelo XGBoost e retorna predições."""
    print(f"\nAvaliando modelo XGBoost...")
    
    import xgboost as xgb
    
    # Criar DMatrix com nomes de features se fornecidos
    if feature_names is not None:
        dtest = xgb.DMatrix(X, feature_names=feature_names)
    else:
        dtest = xgb.DMatrix(X)
    
    probs = model.predict(dtest)
    preds = (probs >= 0.5).astype(int)
    
    print(f"   Predições geradas")
    return preds, probs


def calculate_metrics(y_true, y_pred, y_prob, model_name):
    """Calcula todas as métricas para um modelo."""
    print(f"\nCalculando métricas para: {model_name}")
    
    # Métricas básicas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Métricas avançadas
    roc_auc = roc_auc_score(y_true, y_prob)
    aucpr = average_precision_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Matriz de confusão
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'aucpr': aucpr,
        'mcc': mcc,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }
    
    print(f"   AUCPR: {aucpr:.4f}")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    print(f"   F1: {f1:.4f}")
    
    return metrics


def main():
    print("=" * 80)
    print("GERAÇÃO DE MÉTRICAS CONSOLIDADAS - TODOS OS MODELOS")
    print("=" * 80)
    
    project_root = Path(__file__).parent.parent
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo: {device}")
    
    # Carregar dados de teste
    df_test = load_test_data()
    
    # Preparar dados para ambas as arquiteturas
    X_mlp, y_mlp, features_mlp = prepare_mlp_data(df_test)
    X_xgb, y_xgb, features_xgb = prepare_xgboost_data(df_test)
    
    # Armazenar todas as métricas
    all_metrics = []
    
    # MLP CENTRALIZADO
    mlp_cent_path = project_root / "centralized_training" / "models" / "mlp" / "centralized_model_best.joblib"
    mlp_cent_model = load_mlp_model(mlp_cent_path, device)
    
    if mlp_cent_model is not None:
        preds, probs = evaluate_mlp_model(mlp_cent_model, X_mlp, y_mlp, device)
        metrics = calculate_metrics(y_mlp, preds, probs, "MLP Centralizado")
        all_metrics.append(metrics)
    
    # XGBOOST CENTRALIZADO
    xgb_cent_path = project_root / "centralized_training" / "models" / "xgboost" / "centralized_model_best.joblib"
    xgb_cent_model = load_xgboost_model(xgb_cent_path)
    
    if xgb_cent_model is not None:
        preds, probs = evaluate_xgboost_model(xgb_cent_model, X_xgb, y_xgb, features_xgb)
        metrics = calculate_metrics(y_xgb, preds, probs, "XGBoost Centralizado")
        all_metrics.append(metrics)
    
    # MLP FEDERADO
    mlp_fed_path = project_root / "flwr-mlp" / "models" / "global_model_final.joblib"
    mlp_fed_model = load_mlp_model(mlp_fed_path, device)
    
    if mlp_fed_model is not None:
        preds, probs = evaluate_mlp_model(mlp_fed_model, X_mlp, y_mlp, device)
        metrics = calculate_metrics(y_mlp, preds, probs, "MLP Federado")
        all_metrics.append(metrics)
    
    # XGBOOST FEDERADO - BAGGING
    xgb_bag_path = project_root / "flwr-xgboost" / "models" / "global_model_bagging_final.json"
    xgb_bag_model = load_xgboost_model(xgb_bag_path)
    
    if xgb_bag_model is not None:
        preds, probs = evaluate_xgboost_model(xgb_bag_model, X_xgb, y_xgb, features_xgb)
        metrics = calculate_metrics(y_xgb, preds, probs, "XGBoost Federado (Bagging)")
        all_metrics.append(metrics)
    
    # XGBOOST FEDERADO - CYCLIC
    xgb_cyc_path = project_root / "flwr-xgboost" / "models" / "global_model_cyclic_final.json"
    xgb_cyc_model = load_xgboost_model(xgb_cyc_path)
    
    if xgb_cyc_model is not None:
        preds, probs = evaluate_xgboost_model(xgb_cyc_model, X_xgb, y_xgb, features_xgb)
        metrics = calculate_metrics(y_xgb, preds, probs, "XGBoost Federado (Cyclic)")
        all_metrics.append(metrics)
    
    # SALVAR RESULTADOS
    if len(all_metrics) > 0:
        df_metrics = pd.DataFrame(all_metrics)
        
        # Reordenar colunas para melhor legibilidade
        cols_order = ['model', 'accuracy', 'precision', 'recall', 'f1', 
                      'roc_auc', 'aucpr', 'mcc', 'tp', 'tn', 'fp', 'fn']
        df_metrics = df_metrics[cols_order]
        
        # Salvar em CSV
        output_path = project_root / "test_metrics_all_models.csv"
        df_metrics.to_csv(output_path, index=False, float_format='%.6f')
        
        print("\n" + "=" * 80)
        print("MÉTRICAS CONSOLIDADAS GERADAS!")
        print("=" * 80)
        print(f"\nArquivo salvo: {output_path}")
        print(f"\nResumo:")
        print(df_metrics.to_string(index=False))
        
        print(f"\nRanking por AUCPR:")
        df_ranked = df_metrics.sort_values('aucpr', ascending=False)
        for idx, row in df_ranked.iterrows():
            print(f"   {row['model']:30s} - AUCPR: {row['aucpr']:.6f}")
    else:
        print("\n[ERRO] Nenhum modelo foi avaliado. Verifique se os modelos foram treinados.")


if __name__ == "__main__":
    main()
