"""
Análise SHAP para XGBoost Centralizado.

Analisa importância de features para predição de RainTomorrow usando TreeExplainer.
Usa as mesmas amostras dos outros modelos para comparação justa.
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Adiciona paths do projeto para importar federated_xgboost e utils
sys.path.append(str(Path(__file__).parent.parent / "flwr-xgboost"))
sys.path.append(str(Path(__file__).parent.parent))

from federated_xgboost.task import _encode_wind_directions_cyclic
from utils import (
    load_model_from_joblib,
    load_shap_samples as utils_load_shap_samples,
    create_shap_visualizations,
    save_feature_importance_csv,
    print_top_features,
)

plt.style.use('default')
plt.rcParams.update({'font.size': 10, 'figure.dpi': 100})


def _preprocess_xgb(df_shap: pd.DataFrame):
    """Preprocessamento XGBoost: drop Location, encoding cíclico, imputação mediana."""
    df_local = df_shap.copy()
    df_local = df_local.drop(columns=['Location'], errors='ignore')
    df_local = _encode_wind_directions_cyclic(df_local)
    
    X_df = df_local.drop(columns=['RainTomorrow'])
    y_series = df_local['RainTomorrow']
    feature_names = X_df.columns.tolist()
    
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_np = imputer.fit_transform(X_df.values)
    y_np = y_series.values
    return X_np, y_np, feature_names


def compute_shap_values(model, X, feature_names):
    """Calcula valores SHAP usando TreeExplainer. Lida com outputs binários, multiclasse, sklearn wrappers, etc."""
    print(f"\nCalculando valores SHAP...")
    print(f"   Usando TreeExplainer (otimizado para modelos de árvore)")
    
    try:
        explainer = shap.TreeExplainer(model)
    except Exception as e:
        print(f"Falha inicial ao criar TreeExplainer: {e}")
        try:
            if hasattr(model, 'get_booster') and callable(model.get_booster):
                booster_obj = model.get_booster()
                print("   Booster extraído via model.get_booster()")
                explainer = shap.TreeExplainer(booster_obj)
            else:
                raise
        except Exception as e2:
            raise RuntimeError(f"Não foi possível criar shap.TreeExplainer: {e2}")
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X)
    print(f"   Valores SHAP calculados!")
    
    # Normaliza output para shape (n_samples, n_features)
    if isinstance(shap_values, list):
        if len(shap_values) >= 2:
            shap_values = np.array(shap_values[1])
            print("   SHAP retornou lista por classe, usando classe 1 (Rain)")
        else:
            shap_values = np.array(shap_values[0])
            print("   SHAP retornou lista de 1 classe, usando-a")
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        if shap_values.shape[0] == 2:
            shap_values = shap_values[1]
            print("   SHAP retornou array 3D, usando classe 1 (Rain)")
        else:
            shap_values = shap_values.sum(axis=0)
            print("   SHAP retornou array 3D multiclasse, somando entre classes")
    else:
        shap_values = np.array(shap_values)
    
    print(f"   - Formato final: {shap_values.shape}")
    
    try:
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, tuple, np.ndarray)):
            if len(expected_value) >= 2:
                expected_value = float(expected_value[1])
            else:
                expected_value = float(expected_value[0])
        else:
            expected_value = float(expected_value)
    except Exception:
        expected_value = 0.0
        print("   Não foi possível extrair expected_value, usando 0.0")
    
    print(f"   - Valor base (expected_value): {expected_value:.6f}")
    
    return shap_values, expected_value

def main():
    print("=" * 80)
    print("ANÁLISE SHAP - XGBOOST CENTRALIZADO")
    print("=" * 80)
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    joblib_path = project_root / "centralized_training" / "models" / "xgboost" / "centralized_model_best.joblib"
    data_path = str(project_root / "datasets" / "rain_australia" / "weatherAUS_cleaned.csv")
    shap_indices_path = str(project_root / "datasets" / "shap_sample_indices.csv")
    save_dir = str(script_dir / "shap_results_centralized" / "xgboost")
    
    if not joblib_path.exists():
        print(f"\nERRO: Modelo não encontrado em {joblib_path}")
        print("   Execute o treinamento centralizado do XGBoost primeiro.")
        print("   Comando: python centralized_training/train_centralized_xgboost.py")
        return
    
    model_data = load_model_from_joblib(joblib_path)
    model = model_data['best_model']
    
    X, y, feature_names = utils_load_shap_samples(
        Path(data_path), Path(shap_indices_path), preprocess_fn=_preprocess_xgb
    )
    
    shap_values, base_value = compute_shap_values(model, X, feature_names)
    
    # Salva arrays SHAP para análises
    print(f"\nSalvando arrays SHAP para análises avançadas...")
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    
    np.save(save_dir_path / 'shap_values.npy', shap_values)
    np.save(save_dir_path / 'feature_values.npy', X)
    np.save(save_dir_path / 'target_values.npy', y)
    
    import json
    with open(save_dir_path / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    with open(save_dir_path / 'base_value.json', 'w') as f:
        json.dump({'base_value': float(base_value)}, f, indent=2)
    
    print(f"   Arquivos salvos:")
    print(f"      - shap_values.npy (shape: {shap_values.shape})")
    print(f"      - feature_values.npy (shape: {X.shape})")
    print(f"      - target_values.npy (shape: {y.shape})")
    print(f"      - feature_names.json ({len(feature_names)} features)")
    print(f"      - base_value.json (expected value: {base_value:.4f})")
    
    create_shap_visualizations(shap_values, X, y, feature_names, Path(save_dir), model_name="XGBoost Centralizado")
    df_all, df_0, df_1, df_comparison = save_feature_importance_csv(
        shap_values, X, y, feature_names, Path(save_dir)
    )
    print_top_features(df_all, df_comparison)
    
    print(f"\n" + "=" * 80)
    print("ANÁLISE SHAP CONCLUÍDA!")
    print("=" * 80)
    print(f"\nResultados salvos em: {save_dir}")
    print()


if __name__ == "__main__":
    main()
