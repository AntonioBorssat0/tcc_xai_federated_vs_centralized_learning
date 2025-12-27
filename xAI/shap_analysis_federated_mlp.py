"""
Análise SHAP para MLP Federado.

Gera importâncias de features para RainTomorrow usando GradientExplainer.
Usa as mesmas amostras do centralizado para comparação justa.
"""

import torch
import pandas as pd
import numpy as np
import shap
from pathlib import Path
import sys

# Adiciona paths do projeto para importar utils e federated_mlp
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "flwr-mlp"))

from utils import (
    load_model_from_joblib,
    load_shap_samples as utils_load_shap_samples,
    create_shap_visualizations,
    save_feature_importance_csv,
    print_top_features,
)
from federated_mlp.task import WeatherMLP, prepare_weather_data, LABEL_COL


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
    """Calcula valores SHAP usando GradientExplainer (mesmo padrão do centralizado)."""
    print(f"\nCalculando valores SHAP com GradientExplainer...")
    print(f"   - Amostras: {len(X)}")
    print(f"   - Features: {len(feature_names)}")
    
    # Tensor para device
    X_tensor = torch.FloatTensor(X).to(device)
    
    # Background: amostra aleatória de até 100 instâncias
    background_size = min(100, len(X))
    background_indices = np.random.choice(len(X), background_size, replace=False)
    background = X_tensor[background_indices]
    print(f"   - Background: {len(background)} amostras")
    
    explainer = shap.GradientExplainer(model, background)
    print(f"   - Computando SHAP...")
    shap_values = explainer.shap_values(X_tensor)
    
    # Alguns backends retornam tupla (values, expected), manter apenas os valores
    if isinstance(shap_values, tuple):
        shap_values = shap_values[0]
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 0]
    
    print(f"   SHAP calculado! shape={shap_values.shape}")
    return shap_values


def load_global_model(model_path: str):
    """Carrega o modelo federado global salvo em .pt (legacy)."""
    print(f"Carregando modelo de: {model_path}")
    
    checkpoint = torch.load(model_path, weights_only=False)
    
    base_model = WeatherMLP(
        input_size=checkpoint['input_size'],
        hidden1=checkpoint['hidden1'],
        hidden2=checkpoint['hidden2'],
        hidden3=checkpoint['hidden3'],
        dropout1=checkpoint['dropout1'],
        dropout2=checkpoint['dropout2']
    )
    
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.eval()
    
    model = ModelWrapper(base_model)
    model.eval()
    
    print(f"Modelo carregado com sucesso!")
    print(f"   - Tamanho de entrada: {checkpoint['input_size']}")
    print(f"   - Arquitetura: {checkpoint['hidden1']} → {checkpoint['hidden2']} → {checkpoint['hidden3']}")
    print(f"   - Treinado por {checkpoint['num_rounds']} rounds")
    
    return model, checkpoint

def main():
    print("=" * 80)
    print("ANÁLISE SHAP - MLP FEDERADO")
    print("=" * 80)
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    legacy_model_path = project_root / "flwr-mlp" / "models" / "global_model_final.pt"
    joblib_path = project_root / "flwr-mlp" / "models" / "global_model_final.joblib"
    data_path = str(project_root / "datasets" / "rain_australia" / "weatherAUS_cleaned.csv")
    shap_indices_path = str(project_root / "datasets" / "shap_sample_indices.csv")
    save_dir = str(script_dir / "shap_results_federated" / "mlp")
    
    if not joblib_path.exists() and not legacy_model_path.exists():
        print(f"\nERRO: Modelo não encontrado em {joblib_path} nem {legacy_model_path}")
        print("   Treine o modelo federado primeiro.")
        print("   Dica: padronize o salvamento em JOBLIB no servidor federado.")
        return
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo: {device}")
    scaler = None
    if joblib_path.exists():
        model_data = load_model_from_joblib(joblib_path)
        base_model = model_data['best_model']
        scaler = model_data.get('scaler', None)
        model = ModelWrapper(base_model).to(device)
        model.eval()
    else:
        model, _ = load_global_model(str(legacy_model_path))
        model.to(device)
    
    # Usa o mesmo scaler do treinamento (se disponível no JOBLIB)
    def preprocess_mlp(df_shap: pd.DataFrame):
        df_processed = prepare_weather_data(df_shap.copy(), use_location=False)
        X_df = df_processed.drop(columns=[LABEL_COL])
        y_series = df_processed[LABEL_COL]
        feat_names = X_df.columns.tolist()
        X_np = X_df.values
        y_np = y_series.values
        from sklearn.preprocessing import StandardScaler as _SS
        if scaler is not None:
            X_scaled = scaler.transform(X_np)
        else:
            local_scaler = _SS()
            X_scaled = local_scaler.fit_transform(X_np)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return X_scaled, y_np, feat_names

    X_sample, y_sample, feature_names = utils_load_shap_samples(
        Path(data_path), Path(shap_indices_path), preprocess_fn=preprocess_mlp
    )

    shap_values = compute_shap_values(model, X_sample, feature_names, device)

    # Salva arrays SHAP para análises avançadas
    print(f"\nSalvando arrays SHAP para análises avançadas...")
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    
    np.save(save_dir_path / 'shap_values.npy', shap_values)
    np.save(save_dir_path / 'feature_values.npy', X_sample)
    np.save(save_dir_path / 'target_values.npy', y_sample)
    
    import json
    with open(save_dir_path / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    print(f"   Arquivos salvos:")
    print(f"      - shap_values.npy (shape: {shap_values.shape})")
    print(f"      - feature_values.npy (shape: {X_sample.shape})")
    print(f"      - target_values.npy (shape: {y_sample.shape})")
    print(f"      - feature_names.json ({len(feature_names)} features)")

    create_shap_visualizations(shap_values, X_sample, y_sample, feature_names, Path(save_dir), model_name="MLP Federado")
    df_all, df_0, df_1, df_comparison = save_feature_importance_csv(
        shap_values, X_sample, y_sample, feature_names, Path(save_dir)
    )
    print_top_features(df_all, df_comparison)
    
    print("\n" + "=" * 80)
    print("ANÁLISE SHAP CONCLUÍDA!")
    print("=" * 80)
    print(f"\nResultados salvos em: {Path(save_dir).absolute()}")


if __name__ == "__main__":
    main()
