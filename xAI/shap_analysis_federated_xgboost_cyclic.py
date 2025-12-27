"""
Análise SHAP para XGBoost Federado (Estratégia Cyclic).

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
import tempfile
import os
import json
import re

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


def fix_base_score_in_json(json_path: str) -> str:
    """
    Corrige o campo base_score no arquivo JSON do XGBoost.
    
    Converte formato '[2.2154084E-1]' para '0.22154084'.
    Esse fix é necessário porque o SHAP não consegue parsear arrays como string.
    
    Returns:
        Path para o arquivo JSON corrigido.
    """
    with open(json_path, 'r') as f:
        model_text = f.read()
    
    pattern1 = r'"base_score"\s*:\s*"\[([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\]"'
    
    def replace_base_score(match):
        original = match.group(0)
        value_str = match.group(1)
        value_float = float(value_str)
        replacement = f'"base_score": "{value_float}"'
        print(f"   Corrigido base_score: {original} → {replacement}")
        return replacement
    
    model_text_fixed = re.sub(pattern1, replace_base_score, model_text)
    
    pattern2 = r'"base_score"\s*:\s*\[([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\]'
    
    def replace_base_score_array(match):
        original = match.group(0)
        value_str = match.group(1)
        value_float = float(value_str)
        replacement = f'"base_score": "{value_float}"'
        print(f"   Corrigido base_score (array): {original} → {replacement}")
        return replacement
    
    model_text_fixed = re.sub(pattern2, replace_base_score_array, model_text_fixed)
    
    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w')
    tmpf.write(model_text_fixed)
    tmpf.close()
    
    return tmpf.name


def load_xgboost_model(model_path: str):
    """
    Carrega o modelo XGBoost salvo com correção do base_score.
    
    O modelo cyclic pode ter base_score em formato de array,
    o que causa erro no SHAP. Esta função corrige isso automaticamente.
    
    Returns:
        xgboost.Booster compatível com shap.TreeExplainer.
    """
    print(f"Carregando modelo XGBoost (Cyclic) de: {model_path}")
    
    # sempre corrigir base_score via JSON antes de usar com SHAP
    tmp_json_1 = None
    tmp_json_2 = None
    
    try:
        bst_temp = None
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            if isinstance(model_data, dict) and 'model' in model_data:
                model_bytes = model_data['model']
            else:
                model_bytes = model_data
            
            bst_temp = xgb.Booster()
            bst_temp.load_model(bytearray(model_bytes))
            print(f"Modelo carregado do pickle")
        except Exception as e:
            print(f"Falha ao carregar pickle ({e}), tentando JSON...")
            bst_temp = xgb.Booster()
            bst_temp.load_model(model_path)
            print(f"Modelo carregado do JSON")
        
        # Salva em JSON temporário e aplica fix do base_score
        print(f"   Aplicando correção do base_score...")
        tmp_json_1 = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp_json_1.close()
        bst_temp.save_model(tmp_json_1.name)
        
        tmp_json_2 = fix_base_score_in_json(tmp_json_1.name)
        
        bst = xgb.Booster()
        bst.load_model(tmp_json_2)
        print(f"   Modelo carregado com base_score corrigido")
        
        try:
            print(f"   - Número de features: {bst.num_features()}")
        except Exception:
            pass
        
        # Testa compatibilidade com SHAP
        try:
            _ = shap.TreeExplainer(bst)
            print("   Modelo compatível com shap.TreeExplainer")
            return bst
        except ValueError as e:
            if "could not convert string to float" in str(e):
                raise RuntimeError(
                    f"ERRO: SHAP falhou mesmo após correção do base_score. "
                    f"Pode ser problema de versão do SHAP. Erro: {e}\n"
                    f"   Tente: pip install --upgrade shap"
                )
            else:
                raise RuntimeError(f"shap.TreeExplainer falhou: {e}")
        except Exception as e:
            raise RuntimeError(f"Erro inesperado com shap.TreeExplainer: {e}")
            
    finally:
        for tmp_file in [tmp_json_1, tmp_json_2]:
            if tmp_file is not None:
                tmp_path = tmp_file.name if hasattr(tmp_file, 'name') else tmp_file
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass


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
    """
    Calcula valores SHAP usando TreeExplainer.
    
    Lida com diferentes formatos de saída (binário, multi-classe, sklearn wrappers).
    
    Returns:
        shap_values: array (n_samples, n_features)
        expected_value: valor base para explicações SHAP
    """
    print(f"\nCalculando valores SHAP...")
    print(f"   Usando TreeExplainer (otimizado para modelos de árvore)")
    
    try:
        explainer = shap.TreeExplainer(model)
    except Exception as e:
        print(f"TreeExplainer falhou inicialmente: {e}")
        try:
            if hasattr(model, 'get_booster') and callable(model.get_booster):
                booster_obj = model.get_booster()
                print("   Booster extraído via model.get_booster()")
                explainer = shap.TreeExplainer(booster_obj)
            else:
                raise
        except Exception as e2:
            raise RuntimeError(f"Não foi possível criar shap.TreeExplainer: {e2}")
    
    shap_values = explainer.shap_values(X)
    print(f"   Valores SHAP calculados!")
    
    # Normaliza formato SHAP para (n_samples, n_features)
    if isinstance(shap_values, list):
        if len(shap_values) >= 2:
            shap_values = np.array(shap_values[1])
            print("   SHAP retornou lista por classe → usando classe 1 (Rain)")
        else:
            shap_values = np.array(shap_values[0])
            print("   SHAP retornou lista de classe única → usando ela")
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        if shap_values.shape[0] == 2:
            shap_values = shap_values[1]
            print("   SHAP retornou array 3D → usando classe 1 (Rain)")
        else:
            shap_values = shap_values.sum(axis=0)
            print("   SHAP retornou array 3D multi-classe → somando classes")
    else:
        shap_values = np.array(shap_values)
    
    print(f"   - Shape final: {shap_values.shape}")
    
    # expected_value pode ser array ou float
    try:
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, tuple, np.ndarray)):
            if len(expected_value) >= 2:
                expected_value = float(expected_value[1])
            elif len(expected_value) == 1:
                expected_value = float(expected_value[0])
            else:
                expected_value = 0.0
        else:
            if isinstance(expected_value, (int, float, np.floating)):
                expected_value = float(expected_value)
            else:
                expected_value = 0.0
    except Exception:
        expected_value = 0.0
        print("   Não foi possível extrair expected_value, usando 0.0")
    
    print(f"   - Valor base: {expected_value:.6f}")
    
    return shap_values, expected_value


def main():
    print("=" * 80)
    print("ANÁLISE SHAP - XGBOOST FEDERADO (CYCLIC)")
    print("=" * 80)
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    legacy_model_path = project_root / "flwr-xgboost" / "models" / "global_model_cyclic_final.pt"
    joblib_path = project_root / "flwr-xgboost" / "models" / "global_model_cyclic_final.joblib"
    data_path = str(project_root / "datasets" / "rain_australia" / "weatherAUS_cleaned.csv")
    shap_indices_path = str(project_root / "datasets" / "shap_sample_indices.csv")
    save_dir = str(script_dir / "shap_results_federated" / "xgboost" / "cyclic_strategy")
    
    if not joblib_path.exists() and not legacy_model_path.exists():
        print(f"\nERRO: Modelo não encontrado em {joblib_path} nem {legacy_model_path}")
        print("   Treine o XGBoost federado (Cyclic) primeiro.")
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
    
    create_shap_visualizations(shap_values, X, y, feature_names, Path(save_dir), model_name="XGBoost Federado (Cyclic)")
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
