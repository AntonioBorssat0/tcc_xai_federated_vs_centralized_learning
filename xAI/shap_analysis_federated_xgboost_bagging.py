"""
Análise SHAP para XGBoost Federado (Estratégia Bagging).

Gera importâncias de features para RainTomorrow com TreeExplainer.
Usa as mesmas amostras das outras abordagens para comparação justa.
"""

import xgboost as xgb
import pandas as pd
import numpy as np
import shap
from pathlib import Path
import sys
import pickle

# Adiciona paths do projeto para importar utils e federated_xgboost
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "flwr-xgboost"))


from utils import (
    load_model_from_joblib,
    load_shap_samples as utils_load_shap_samples,
    create_shap_visualizations,
    save_feature_importance_csv,
    print_top_features,
)
from federated_xgboost.task import _encode_wind_directions_cyclic

def load_xgboost_model(model_path: str):
    """Carrega o modelo XGBoost salvo (formato legacy .pt)."""
    print(f"Carregando modelo XGBoost (Bagging) de: {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        if isinstance(model_data, dict) and 'model' in model_data:
            model_bytes = model_data['model']
        else:
            model_bytes = model_data
        
        bst = xgb.Booster()
        bst.load_model(bytearray(model_bytes))
        
        print(f"Modelo carregado com sucesso (pickle)!")
    except:
        bst = xgb.Booster()
        bst.load_model(model_path)
        print(f"Modelo carregado com sucesso (JSON)!")
    
    print(f"   - Número de features: {bst.num_features()}")
    
    return bst


def _preprocess_xgb(df_shap: pd.DataFrame):
    """Preprocessamento para XGBoost (mesmo do treino, com imputação mediana)."""
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
    """Calcula valores SHAP usando TreeExplainer."""
    print(f"\nCalculando valores SHAP...")
    print(f"   Usando TreeExplainer (otimizado para modelos de árvore)")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    print(f"   Valores SHAP calculados!")
    print(f"   - Shape: {shap_values.shape}")
    
    # expected_value pode ser array ou float
    expected_value = explainer.expected_value
    if isinstance(expected_value, np.ndarray):
        expected_value = float(expected_value[0]) if len(expected_value) > 0 else 0.0
    
    print(f"   - Valor base: {expected_value:.4f}")
    
    return shap_values, expected_value


def main():
    print("=" * 80)
    print("ANÁLISE SHAP - XGBOOST FEDERADO (BAGGING)")
    print("=" * 80)
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    legacy_model_path = project_root / "flwr-xgboost" / "models" / "global_model_bagging_final.pt"
    joblib_path = project_root / "flwr-xgboost" / "models" / "global_model_bagging_final.joblib"
    data_path = str(project_root / "datasets" / "rain_australia" / "weatherAUS_cleaned.csv")
    shap_indices_path = str(project_root / "datasets" / "shap_sample_indices.csv")
    save_dir = str(script_dir / "shap_results_federated" / "xgboost" / "bagging_strategy")
    
    if not joblib_path.exists() and not legacy_model_path.exists():
        print(f"\nERRO: Modelo não encontrado em {joblib_path} nem {legacy_model_path}")
        print("   Treine o XGBoost federado (Bagging) primeiro.")
        return

    if joblib_path.exists():
        model_data = load_model_from_joblib(joblib_path)
        model = model_data['best_model']
    else:
        model = load_xgboost_model(str(legacy_model_path))

    X, y, feature_names = utils_load_shap_samples(
        Path(data_path), Path(shap_indices_path), preprocess_fn=_preprocess_xgb
    )

    shap_values, base_value = compute_shap_values(model, X, feature_names)

    # Salva arrays SHAP para análises avançadas
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
    print(f"      - base_value.json (valor esperado: {base_value:.4f})")

    create_shap_visualizations(shap_values, X, y, feature_names, Path(save_dir), model_name="XGBoost Federado (Bagging)")
    df_all, df_0, df_1, df_comparison = save_feature_importance_csv(
        shap_values, X, y, feature_names, Path(save_dir)
    )
    print_top_features(df_all, df_comparison)
    
    print(f"\n" + "=" * 80)
    print("ANÁLISE SHAP CONCLUÍDA!")
    print("=" * 80)
    print(f"\nResultados salvos em: {save_dir}")


if __name__ == "__main__":
    main()
