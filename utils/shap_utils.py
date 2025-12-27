"""Utilitários de Análise SHAP para o Projeto.
Funções compartilhadas para visualização e análise SHAP.
"""

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


def load_model_from_joblib(joblib_path: Path) -> Dict[str, Any]:
    """
    Carrega modelo de arquivo joblib padronizado.
    
    Returns:
        Dicionário com model, scaler, feature_names, metadata, etc.
    """
    print(f"Carregando modelo de: {joblib_path}")
    
    if not joblib_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {joblib_path}")
    
    model_data = joblib.load(joblib_path)
    
    print(f"   Modelo carregado com sucesso!")
    
    # Mostrar metadados se disponíveis
    if 'metadata' in model_data:
        meta = model_data['metadata']
        print(f"   - Tipo: {meta.get('model_type', 'N/A')}")
        print(f"   - Framework: {meta.get('framework', 'N/A')}")
        if 'accuracy' in meta:
            print(f"   - Accuracy: {meta['accuracy']:.4f}")
        if 'auc' in meta:
            print(f"   - AUC: {meta['auc']:.4f}")
    
    return model_data


def load_shap_samples(
    data_path: Path,
    shap_indices_path: Path,
    preprocess_fn: callable,
    feature_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Carrega amostras FIXAS para SHAP (padronizadas entre todas as abordagens).
    
    Args:
        data_path: Caminho para dataset completo
        shap_indices_path: Caminho para CSV com índices das amostras SHAP
        preprocess_fn: Função para preprocessar dados (retorna X, y, feature_names)
        feature_names: Lista opcional de nomes de features esperados
    
    Returns:
        X: Features preprocessadas (numpy array)
        y: Labels (numpy array)
        feature_names: Lista de nomes das features
    """
    print("\nCarregando dados para análise SHAP...")
    
    # Carregar dataset completo
    df = pd.read_csv(data_path)
    print(f"   - Total de amostras: {len(df):,}")
    
    # Carregar índices das amostras SHAP
    shap_indices_df = pd.read_csv(shap_indices_path)
    shap_indices = shap_indices_df['index'].values
    
    print(f"   Carregando amostra FIXA: {len(shap_indices):,} amostras")
    print(f"   [INFO] MESMAS amostras usadas em todas as abordagens!")
    
    # Filtrar dataset para amostras SHAP
    df_shap = df.iloc[shap_indices].copy()
    
    # Preprocessar dados
    X, y, feat_names = preprocess_fn(df_shap)
    
    print(f"   Features: {len(feat_names)}")
    print(f"   Amostras: {len(X):,}")
    
    # Mostrar distribuição de classes
    class_0 = np.sum(y == 0)
    class_1 = np.sum(y == 1)
    print(f"\n   Distribuição de classes:")
    print(f"      Classe 0 (No Rain): {class_0:,} ({class_0/len(y)*100:.1f}%)")
    print(f"      Classe 1 (Rain)   : {class_1:,} ({class_1/len(y)*100:.1f}%)")
    
    return X, y, feat_names


def create_shap_visualizations(
    shap_values: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    save_dir: Path,
    model_name: str
) -> None:
    """
    Cria visualizações SHAP padronizadas (4 figuras essenciais).
    
    Args:
        shap_values: Array de valores SHAP (n_samples, n_features)
        X: Valores das features (n_samples, n_features)
        y: Labels (n_samples,)
        feature_names: Lista de nomes das features
        save_dir: Diretório para salvar as figuras
        model_name: Nome do modelo (usado nos títulos)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCriando visualizações SHAP...")
    print(f"   Salvando em: {save_dir}")
    
    # Separar por classe
    class_0_mask = y == 0
    class_1_mask = y == 1
    
    shap_values_class_0 = shap_values[class_0_mask]
    shap_values_class_1 = shap_values[class_1_mask]
    X_class_0 = X[class_0_mask]
    X_class_1 = X[class_1_mask]
    
    print(f"   - Classe 0 (No Rain): {len(shap_values_class_0):,} amostras")
    print(f"   - Classe 1 (Rain)   : {len(shap_values_class_1):,} amostras")
    
    # FIGURA 1: Feature Importance (TODAS)
    print(f"\n   [1/4] Feature Importance (todas as classes)...")
    
    shap_explanation = shap.Explanation(
        values=shap_values,
        base_values=np.zeros(len(shap_values)),
        data=X,
        feature_names=feature_names
    )
    
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_explanation, max_display=19, show=False)
    
    # Cor azul padrão SHAP
    ax = plt.gca()
    shap_blue = (0.0, 0.54337757, 0.98337906)
    for bar in ax.patches:
        bar.set_color(shap_blue)
        bar.set_alpha(0.9)
    
    for text in ax.texts:
        text.set_color(shap_blue)
        text.set_fontweight('bold')
    
    plt.title(f"Feature Importance - Todas as Classes ({model_name})", 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / "feature_importance_all.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      Salvo: feature_importance_all.png")
    
    # FIGURA 2: Beeswarm Plot (CLASSE 1 - Rain)
    print(f"\n   [2/4] Beeswarm Plot (Classe 1 - Rain)...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values_class_1, X_class_1, feature_names=feature_names,
        show=False, max_display=19
    )
    plt.title(f"SHAP Values - Classe 1 (Rain) - {model_name}", 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / "beeswarm_class_1.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      Salvo: beeswarm_class_1.png")
    
    # FIGURA 3: Waterfall Plot (CLASSE 1 - Exemplo)
    print(f"\n   [3/4] Waterfall Plot (Classe 1 - Amostra 1)...")
    if len(shap_values_class_1) > 0:
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_class_1[0],
                base_values=0,
                data=X_class_1[0],
                feature_names=feature_names
            ),
            show=False,
            max_display=15
        )
        plt.title(f"SHAP Waterfall - Classe 1 Amostra 1 ({model_name})", 
                  fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_dir / "waterfall_class_1_sample_1.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      Salvo: waterfall_class_1_sample_1.png")
    else:
        print(f"      [AVISO] Sem amostras da Classe 1 para waterfall plot")
    
    # FIGURA 4: Comparação (CLASSE 0 vs CLASSE 1)
    print(f"\n   [4/4] Gráfico de Comparação (Classe 0 vs Classe 1)...")
    
    mean_shap_0 = np.abs(shap_values_class_0).mean(axis=0)
    mean_shap_1 = np.abs(shap_values_class_1).mean(axis=0)
    
    # Ordenar por importância da Classe 1 (decrescente)
    sorted_idx = np.argsort(mean_shap_1)[::-1]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(feature_names))
    width = 0.35
    
    bars1 = ax.barh(y_pos - width/2, mean_shap_0[sorted_idx], width, 
                    label='Classe 0 (No Rain)', color='skyblue', alpha=0.8)
    bars2 = ax.barh(y_pos + width/2, mean_shap_1[sorted_idx], width, 
                    label='Classe 1 (Rain)', color='salmon', alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.invert_yaxis()
    ax.set_xlabel('Mean Absolute SHAP Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.set_title(f'Comparação de Importância: No Rain vs Rain ({model_name})', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "comparison_class_0_vs_1.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      Salvo: comparison_class_0_vs_1.png")
    
    print(f"\n   Todas as 4 figuras criadas com sucesso!")


def save_feature_importance_csv(
    shap_values: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    save_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Salva importância das features em arquivos CSV (padronizado).
    
    Returns:
        df_all: Importância geral
        df_0: Importância Classe 0
        df_1: Importância Classe 1
        df_comparison: Comparação entre classes
    """
    save_dir = Path(save_dir)
    
    print(f"\nSalvando importância das features em CSV...")
    
    # Importância geral
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    df_all = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values('Mean_Abs_SHAP', ascending=False)
    df_all.to_csv(save_dir / "feature_importance_all.csv", index=False)
    print(f"   feature_importance_all.csv")
    
    # Importância específica por classe
    class_0_mask = y == 0
    class_1_mask = y == 1
    
    mean_abs_shap_0 = np.abs(shap_values[class_0_mask]).mean(axis=0)
    mean_abs_shap_1 = np.abs(shap_values[class_1_mask]).mean(axis=0)
    
    df_0 = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap_0
    }).sort_values('Mean_Abs_SHAP', ascending=False)
    df_0.to_csv(save_dir / "feature_importance_class_0.csv", index=False)
    print(f"   feature_importance_class_0.csv")
    
    df_1 = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap_1
    }).sort_values('Mean_Abs_SHAP', ascending=False)
    df_1.to_csv(save_dir / "feature_importance_class_1.csv", index=False)
    print(f"   feature_importance_class_1.csv")
    
    # Comparação
    df_comparison = pd.DataFrame({
        'Feature': feature_names,
        'Class_0_SHAP': mean_abs_shap_0,
        'Class_1_SHAP': mean_abs_shap_1,
        'Difference': mean_abs_shap_1 - mean_abs_shap_0,
        'Abs_Difference': np.abs(mean_abs_shap_1 - mean_abs_shap_0)
    }).sort_values('Abs_Difference', ascending=False)
    df_comparison.to_csv(save_dir / "comparison_class_0_vs_1.csv", index=False)
    print(f"   comparison_class_0_vs_1.csv")
    
    print(f"   Todos os CSVs salvos!")
    
    return df_all, df_0, df_1, df_comparison


def print_top_features(df_all: pd.DataFrame, df_comparison: pd.DataFrame, n: int = 5):
    """Mostra as top features em formato padronizado."""
    print(f"\nTOP {n} FEATURES:")
    
    print(f"\n   Geral (todas as classes):")
    for idx, (_, row) in enumerate(df_all.head(n).iterrows(), 1):
        print(f"      {idx}. {row['Feature']:25s} : {row['Mean_Abs_SHAP']:.4f}")
    
    print(f"\n   Maior diferença entre classes (Classe 1 vs Classe 0):")
    for idx, (_, row) in enumerate(df_comparison.head(n).iterrows(), 1):
        diff = row['Difference']
        sign = "+" if diff > 0 else "-"
        print(f"      {idx}. {row['Feature']:25s} : Diff = {diff:+.4f} [{sign}]")
