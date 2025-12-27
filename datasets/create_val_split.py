"""
Script para criar divisão 70/10/20 (train/val/test) a partir do dataset completo.
Recria os conjuntos de treino e validação mantendo o conjunto de teste fixo.
Mantém estratificação por Location + RainTomorrow para garantir representatividade.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

print("=" * 80)
print("CRIAÇÃO DE DIVISÃO 70/10/20 (TRAIN/VAL/TEST)")
print("=" * 80)

# Configuração de seed para reprodutibilidade
RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.10
TEST_SIZE = 0.20

# Definição de caminhos
script_dir = Path(__file__).parent
data_path = script_dir / "rain_australia" / "weatherAUS_cleaned.csv"
train_indices_path = script_dir / "train_indices.csv"
val_indices_path = script_dir / "val_indices.csv"
test_indices_path = script_dir / "test_indices.csv"

print(f"\nCarregando dataset completo...")
df_full = pd.read_csv(data_path)
print(f"   Total de amostras: {len(df_full):,}")

print(f"\nCarregando índices de teste (fixos - não mudam)...")
test_indices_df = pd.read_csv(test_indices_path)
test_idx = test_indices_df['index'].values
print(f"   Test: {len(test_idx):,} amostras (20.0% - mantido)")

# Extração dos dados que não estão no conjunto de teste
non_test_mask = ~df_full.index.isin(test_idx)
df_trainval = df_full[non_test_mask].copy()

print(f"   Train+Val disponíveis: {len(df_trainval):,} amostras (80.0%)")

# Análise da distribuição dos dados disponíveis
print(f"\nDistribuição dos dados disponíveis para train+val:")
class_dist = df_trainval['RainTomorrow'].value_counts()
print(f"   Sem Chuva (0): {class_dist[0]:,} ({class_dist[0]/len(df_trainval)*100:.1f}%)")
print(f"   Chuva (1): {class_dist[1]:,} ({class_dist[1]/len(df_trainval)*100:.1f}%)")
print(f"   Locations: {df_trainval['Location'].nunique()}")

# Criação de coluna combinada para estratificação
print(f"\nCriando estratificação por Location + RainTomorrow...")
df_trainval['stratify_col'] = df_trainval['Location'].astype(str) + "_" + df_trainval['RainTomorrow'].astype(str)

# Verificação de amostras mínimas por estrato
strat_counts = df_trainval['stratify_col'].value_counts()
min_samples = strat_counts.min()
print(f"   Combinações únicas: {len(strat_counts)}")
print(f"   Mínimo de amostras por combinação: {min_samples}")

if min_samples < 2:
    print(f"   Algumas combinações têm < 2 amostras")
    print(f"      Usando apenas RainTomorrow para estratificação...")
    stratify_column = df_trainval['RainTomorrow']
else:
    stratify_column = df_trainval['stratify_col']

# Cálculo da proporção para divisão train/val
val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)

print(f"\nDividindo em Train (70% dataset) e Val (10% dataset)...")
print(f"   Proporção usada: {(1-val_ratio)*100:.1f}% train / {val_ratio*100:.1f}% val")

train_idx_new, val_idx_new = train_test_split(
    df_trainval.index,
    test_size=val_ratio,
    random_state=RANDOM_STATE,
    stratify=stratify_column
)

print(f"   Train: {len(train_idx_new):,} amostras")
print(f"   Val: {len(val_idx_new):,} amostras")

# Validação da estratificação
train_df_new = df_trainval.loc[train_idx_new]
val_df_new = df_trainval.loc[val_idx_new]

print(f"\nVerificando estratificação...")

# Análise da distribuição de classes
train_class_dist = train_df_new['RainTomorrow'].value_counts(normalize=True)
val_class_dist = val_df_new['RainTomorrow'].value_counts(normalize=True)

print(f"   Distribuição de classes (treino novo):")
print(f"      Sem Chuva: {train_class_dist[0]*100:.1f}%")
print(f"      Chuva: {train_class_dist[1]*100:.1f}%")

print(f"   Distribuição de classes (validação):")
print(f"      Sem Chuva: {val_class_dist[0]*100:.1f}%")
print(f"      Chuva: {val_class_dist[1]*100:.1f}%")

print(f"\n   Locations no treino: {train_df_new['Location'].nunique()}")
print(f"   Locations na validação: {val_df_new['Location'].nunique()}")

# Salvamento dos índices de validação
print(f"\nSalvando índices de validação...")

val_indices_df = pd.DataFrame({
    'index': val_idx_new,
    'Location': df_full.loc[val_idx_new, 'Location'].values,
    'RainTomorrow': df_full.loc[val_idx_new, 'RainTomorrow'].values
})
val_indices_df.to_csv(val_indices_path, index=False)
print(f"   {val_indices_path.name}")

# Atualização do arquivo train_indices.csv com os novos índices
train_indices_df_new = pd.DataFrame({
    'index': train_idx_new,
    'Location': df_full.loc[train_idx_new, 'Location'].values,
    'RainTomorrow': df_full.loc[train_idx_new, 'RainTomorrow'].values
})
train_indices_df_new.to_csv(train_indices_path, index=False)
print(f"   {train_indices_path.name} (atualizado)")

print(f"\n" + "=" * 80)
print("DIVISÃO TRAIN/VAL CRIADA COM SUCESSO!")
print("=" * 80)
print(f"\nResumo:")
total_samples = len(df_full)
print(f"   Dataset total: {total_samples:,} amostras")
print(f"   Train: {len(train_idx_new):,} amostras ({len(train_idx_new)/total_samples*100:.1f}%)")
print(f"   Val: {len(val_idx_new):,} amostras ({len(val_idx_new)/total_samples*100:.1f}%)")
print(f"   Test: {len(test_idx):,} amostras ({len(test_idx)/total_samples*100:.1f}%)")
print(f"   Random state: {RANDOM_STATE}")
print(f"\nDivisão final: {len(train_idx_new)/total_samples*100:.0f}/{len(val_idx_new)/total_samples*100:.0f}/{len(test_idx)/total_samples*100:.0f} (train/val/test)")
