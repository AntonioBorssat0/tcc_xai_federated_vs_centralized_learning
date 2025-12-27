"""
Gera gráfico comparativo de importância global SHAP para os 5 modelos.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

BASE_PATH = Path(__file__).parent / "shap_results_centralized"
FED_PATH = Path(__file__).parent / "shap_results_federated"

paths = {
    'MLP Centralizado': BASE_PATH / "mlp" / "feature_importance_all.csv",
    'XGBoost Centralizado': BASE_PATH / "xgboost" / "feature_importance_all.csv",
    'MLP Federado (FedAvg)': FED_PATH / "mlp" / "feature_importance_all.csv",
    'XGBoost Fed. (Bagging)': FED_PATH / "xgboost" / "bagging_strategy" / "feature_importance_all.csv",
    'XGBoost Fed. (Cyclic)': FED_PATH / "xgboost" / "cyclic_strategy" / "feature_importance_all.csv",
}

all_data = {}
for model_name, path in paths.items():
    df = pd.read_csv(path)
    if 'Mean_Abs_SHAP' in df.columns:
        df = df.rename(columns={'Mean_Abs_SHAP': 'Importance'})
    elif 'mean_abs_shap' in df.columns:
        df = df.rename(columns={'mean_abs_shap': 'Importance'})
    elif 'importance' in df.columns:
        df = df.rename(columns={'importance': 'Importance'})
    all_data[model_name] = df

top_features = all_data['XGBoost Centralizado'].nlargest(19, 'Importance')['Feature'].tolist()

comparison_data = []
for model_name, df in all_data.items():
    for feature in top_features:
        importance = df[df['Feature'] == feature]['Importance'].values
        if len(importance) > 0:
            comparison_data.append({
                'Modelo': model_name,
                'Feature': feature,
                'Importance': importance[0]
            })

df_comparison = pd.DataFrame(comparison_data)

pivot_data = df_comparison.pivot(index='Feature', columns='Modelo', values='Importance')
feature_order = pivot_data.mean(axis=1).sort_values(ascending=False).index
pivot_data = pivot_data.loc[feature_order]

column_order = [
    'XGBoost Centralizado',
    'MLP Centralizado', 
    'MLP Federado (FedAvg)',
    'XGBoost Fed. (Bagging)',
    'XGBoost Fed. (Cyclic)'
]
pivot_data = pivot_data[column_order]

# FIGURA 1: Heatmap de comparação (todas as features)
fig1, ax1 = plt.subplots(figsize=(12, 12))

sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap="Blues", 
            ax=ax1, cbar_kws={'label': 'SHAP Importance', 'shrink': 0.8},
            linewidths=0.8, linecolor='gray', annot_kws={'fontsize': 20, 'weight': 'bold'})

cbar = ax1.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
cbar.set_label('SHAP Importance', fontsize=20, fontweight='bold')
# ax1.set_title('Comparação de Importância SHAP entre Modelos', 
#               fontsize=28, fontweight='bold', pad=25)
ax1.set_xlabel('Modelo', fontsize=20, fontweight='bold')
ax1.set_ylabel('Feature', fontsize=20, fontweight='bold')
ax1.tick_params(axis='x', rotation=70, labelsize=20)
ax1.tick_params(axis='y', labelsize=20)

plt.tight_layout()

output_path_heatmap = Path(__file__).parent / "shap_global_comparison_heatmap.png"
plt.savefig(output_path_heatmap, dpi=300, bbox_inches='tight')
print(f"   Figura 1 (Heatmap) salva em: {output_path_heatmap}")

output_pdf_heatmap = Path(__file__).parent / "shap_global_comparison_heatmap.pdf"
plt.savefig(output_pdf_heatmap, format='pdf', bbox_inches='tight')
print(f"   Figura 1 PDF salva em: {output_pdf_heatmap}")

plt.close()

# FIGURA 2: Top 10 Features - Barras Horizontais Agrupadas

top_10_features = feature_order[:10]
df_top10 = df_comparison[df_comparison['Feature'].isin(top_10_features)]

fig2, ax2 = plt.subplots(figsize=(10, 6))

y_pos = np.arange(len(top_10_features))
bar_height = 0.15

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
model_order = column_order

top_10_features_reversed = list(reversed(top_10_features))

for i, model in enumerate(model_order):
    model_data = []
    for feature in top_10_features_reversed:
        val = df_top10[(df_top10['Modelo'] == model) & (df_top10['Feature'] == feature)]['Importance'].values
        model_data.append(val[0] if len(val) > 0 else 0)
    
    ax2.barh(y_pos + i * bar_height, model_data, bar_height, 
             label=model, color=colors[i], alpha=0.85, edgecolor='black', linewidth=0.5)

ax2.set_xlabel('SHAP Importance (média |SHAP value|)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Feature', fontsize=13, fontweight='bold')
# ax2.set_title('Top 10 Features por Importância SHAP\n(Comparação entre Modelos)', 
#               fontsize=16, fontweight='bold', pad=20)
ax2.set_yticks(y_pos + bar_height * 2)
ax2.set_yticklabels(top_10_features_reversed, fontsize=13)
ax2.tick_params(axis='x', labelsize=13)
ax2.legend(loc='lower right', fontsize=13, framealpha=0.9)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.set_axisbelow(True)

plt.tight_layout()

output_path_top10 = Path(__file__).parent / "shap_global_comparison_top10.png"
plt.savefig(output_path_top10, dpi=300, bbox_inches='tight')
print(f"   Figura 2 (Top 10) salva em: {output_path_top10}")

output_pdf_top10 = Path(__file__).parent / "shap_global_comparison_top10.pdf"
plt.savefig(output_pdf_top10, format='pdf', bbox_inches='tight')
print(f"   Figura 2 PDF salva em: {output_pdf_top10}")

plt.close()

print("\n" + "="*80)
print("ESTATÍSTICAS DE IMPORTÂNCIA SHAP")
print("="*80)

avg_importance = df_comparison.groupby('Feature')['Importance'].mean().sort_values(ascending=False)
print("\nTop 5 Features (média entre todos os modelos):")
for i, (feature, importance) in enumerate(avg_importance.head(5).items(), 1):
    print(f"{i}. {feature:20s} → {importance:.4f}")

print("\nConsistência entre modelos (Coeficiente de Variação):")
cv_data = df_comparison.groupby('Feature')['Importance'].agg(['mean', 'std'])
cv_data['CV'] = cv_data['std'] / cv_data['mean']
cv_data_sorted = cv_data.sort_values('CV')

print("\nFeatures mais consistentes (baixo CV):")
for i, (feature, row) in enumerate(cv_data_sorted.head(5).iterrows(), 1):
    print(f"{i}. {feature:20s} → CV = {row['CV']:.3f}")

print("\nFeatures mais variáveis (alto CV):")
for i, (feature, row) in enumerate(cv_data_sorted.tail(5).iterrows(), 1):
    print(f"{i}. {feature:20s} → CV = {row['CV']:.3f}")

print("\nCorrelação entre modelos centralizados e federados:")
xgb_central = all_data['XGBoost Centralizado'].set_index('Feature')['Importance']
xgb_bagging = all_data['XGBoost Fed. (Bagging)'].set_index('Feature')['Importance']
xgb_cyclic = all_data['XGBoost Fed. (Cyclic)'].set_index('Feature')['Importance']

from scipy.stats import spearmanr
corr_bagging, p_bagging = spearmanr(xgb_central, xgb_bagging)
corr_cyclic, p_cyclic = spearmanr(xgb_central, xgb_cyclic)

print(f"XGBoost Central vs Bagging: ρ = {corr_bagging:.4f} (p = {p_bagging:.4e})")
print(f"XGBoost Central vs Cyclic:  ρ = {corr_cyclic:.4f} (p = {p_cyclic:.4e})")

mlp_central = all_data['MLP Centralizado'].set_index('Feature')['Importance']
mlp_fed = all_data['MLP Federado (FedAvg)'].set_index('Feature')['Importance']
corr_mlp, p_mlp = spearmanr(mlp_central, mlp_fed)
print(f"MLP Central vs Federado:    ρ = {corr_mlp:.4f} (p = {p_mlp:.4e})")

print("\n" + "="*80)
print("Análise concluída! Use a figura PNG para o documento e PDF para LaTeX.")
print("="*80)
