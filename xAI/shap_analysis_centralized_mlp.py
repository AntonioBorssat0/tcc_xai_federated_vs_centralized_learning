"""
Análise SHAP para MLP Centralizado.

Analisa importância de features para predição de RainTomorrow (classes 0 e 1).
Usa as MESMAS amostras do modelo federado para comparação justa.
"""

import torch
import pandas as pd
import numpy as np
import shap
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler

# Adiciona paths do projeto para importar federated_mlp e utils
sys.path.append(str(Path(__file__).parent.parent / "flwr-mlp"))
sys.path.append(str(Path(__file__).parent.parent))

from federated_mlp.task import WeatherMLP, prepare_weather_data, LABEL_COL
from utils import (
    load_model_from_joblib,
    load_shap_samples as utils_load_shap_samples,
    create_shap_visualizations,
    save_feature_importance_csv,
    print_top_features,
)


class ModelWrapper(torch.nn.Module):
    """Wrapper pra garantir que output do modelo tenha shape correto pro SHAP."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        output = self.model(x)
        if output.dim() == 1:
            output = output.unsqueeze(-1)
        return output


def compute_shap_values(model, X, feature_names, device):
    """Calcula valores SHAP usando GradientExplainer."""
    print(f"\nComputando valores SHAP usando GradientExplainer...")
    print(f"   - Samples: {len(X)}")
    print(f"   - Features: {len(feature_names)}")
    
    X_tensor = torch.FloatTensor(X).to(device)
    
    # Background dataset: amostra aleatória de 100 instâncias
    background_size = min(100, len(X))
    background_indices = np.random.choice(len(X), background_size, replace=False)
    background = X_tensor[background_indices]
    
    print(f"   - Background samples: {len(background)}")
    
    explainer = shap.GradientExplainer(model, background)
    
    print(f"   - Computing SHAP values... (this may take a few minutes)")
    shap_values = explainer.shap_values(X_tensor)
    
    # Squeeze pra obter shape (n_samples, n_features)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 0]
    
    print(f"   Valores SHAP computados!")
    print(f"      Shape: {shap_values.shape}")
    
    return shap_values

def main():
    print("=" * 80)
    print("ANÁLISE SHAP - MLP CENTRALIZADO")
    print("=" * 80)
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    joblib_path = project_root / "centralized_training" / "models" / "mlp" / "centralized_model_best.joblib"
    data_path = str(project_root / "datasets" / "rain_australia" / "weatherAUS_cleaned.csv")
    shap_indices_path = str(project_root / "datasets" / "shap_sample_indices.csv")
    save_dir = str(script_dir / "shap_results_centralized" / "mlp")
    
    if not joblib_path.exists():
        print(f"\nERRO: Modelo não encontrado em {joblib_path}")
        print("   Execute o treinamento centralizado primeiro.")
        print("   Comando: python centralized_training/train_centralized_mlp.py")
        return
    
    model_data = load_model_from_joblib(joblib_path)
    base_model = model_data['best_model']
    scaler = model_data.get('scaler', None)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo: {device}")
    model = ModelWrapper(base_model).to(device)
    model.eval()
    
    # Usa o mesmo scaler do treinamento para preprocessar os dados
    def preprocess_mlp(df_shap: pd.DataFrame):
        df_processed = prepare_weather_data(df_shap.copy(), use_location=False)
        X_df = df_processed.drop(columns=[LABEL_COL])
        y_series = df_processed[LABEL_COL]
        feature_names_local = X_df.columns.tolist()
        X_np = X_df.values
        y_np = y_series.values
        if scaler is not None:
            X_scaled = scaler.transform(X_np)
        else:
            local_scaler = StandardScaler()
            X_scaled = local_scaler.fit_transform(X_np)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return X_scaled, y_np, feature_names_local

    X, y, feature_names = utils_load_shap_samples(
        Path(data_path), Path(shap_indices_path), preprocess_fn=preprocess_mlp
    )
    
    shap_values = compute_shap_values(model, X, feature_names, device)
    
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
    
    print(f"   Arquivos salvos:")
    print(f"      - shap_values.npy (shape: {shap_values.shape})")
    print(f"      - feature_values.npy (shape: {X.shape})")
    print(f"      - target_values.npy (shape: {y.shape})")
    print(f"      - feature_names.json ({len(feature_names)} features)")
    
    create_shap_visualizations(shap_values, X, y, feature_names, Path(save_dir), model_name="MLP Centralizado")
    
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
