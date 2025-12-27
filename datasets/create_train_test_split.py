"""
Script para criar divisão estratificada de treino/teste.
Garante que os modelos centralizado e federado usem exatamente os mesmos dados,
permitindo comparação justa entre as abordagens.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

print("=" * 80)
print("CRIAÇÃO DE DIVISÃO TRAIN/TEST ESTRATIFICADA")
print("=" * 80)

# Configuração de seed para reprodutibilidade
RANDOM_STATE = 42
TEST_SIZE = 0.20
SHAP_SAMPLES = 2000

# Definição de caminhos
script_dir = Path(__file__).parent
data_path = script_dir / "rain_australia" / "weatherAUS_cleaned.csv"
train_indices_path = script_dir / "train_indices.csv"
test_indices_path = script_dir / "test_indices.csv"
shap_indices_path = script_dir / "shap_sample_indices.csv"

print(f"\nCarregando dataset...")
print(f"   {data_path}")

df = pd.read_csv(data_path)
print(f"   Total de amostras: {len(df):,}")

# Verificação da distribuição dos dados
locations = df['Location'].unique()
print(f"   Locations: {len(locations)}")
print(f"   Features: {len(df.columns)}")

# Análise da distribuição de classes
class_dist = df['RainTomorrow'].value_counts()
print(f"\nDistribuição de classes:")
print(f"   Sem Chuva (0): {class_dist[0]:,} ({class_dist[0]/len(df)*100:.1f}%)")
print(f"   Chuva (1): {class_dist[1]:,} ({class_dist[1]/len(df)*100:.1f}%)")

# Criação de coluna combinada para estratificação (Location + RainTomorrow)
print(f"\nCriando estratificação por Location + RainTomorrow...")
df['stratify_col'] = df['Location'].astype(str) + "_" + df['RainTomorrow'].astype(str)

# Verificação de amostras mínimas por estrato
strat_counts = df['stratify_col'].value_counts()
min_samples = strat_counts.min()
if min_samples < 2:
    print(f"   AVISO: Algumas combinações têm apenas {min_samples} amostra(s)")
    print(f"      Removendo combinações com < 2 amostras...")
    valid_strat = strat_counts[strat_counts >= 2].index
    df = df[df['stratify_col'].isin(valid_strat)]
    print(f"      Amostras após filtro: {len(df):,}")

# Divisão estratificada dos dados
print(f"\nDividindo em {int((1-TEST_SIZE)*100)}% treino / {int(TEST_SIZE*100)}% teste...")

train_idx, test_idx = train_test_split(
    df.index,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df['stratify_col']
)

print(f"   Treino: {len(train_idx):,} amostras ({len(train_idx)/len(df)*100:.1f}%)")
print(f"   Teste: {len(test_idx):,} amostras ({len(test_idx)/len(df)*100:.1f}%)")

# Validação da estratificação
train_df = df.loc[train_idx]
test_df = df.loc[test_idx]

print(f"\nVerificando estratificação...")

# Análise da distribuição de classes
train_class_dist = train_df['RainTomorrow'].value_counts(normalize=True)
test_class_dist = test_df['RainTomorrow'].value_counts(normalize=True)

print(f"   Distribuição de classes (treino):")
print(f"      Sem Chuva: {train_class_dist[0]*100:.1f}%")
print(f"      Chuva: {train_class_dist[1]*100:.1f}%")

print(f"   Distribuição de classes (teste):")
print(f"      Sem Chuva: {test_class_dist[0]*100:.1f}%")
print(f"      Chuva: {test_class_dist[1]*100:.1f}%")

# Análise de distribuição por location
print(f"\n   Locations no treino: {train_df['Location'].nunique()}")
print(f"   Locations no teste: {test_df['Location'].nunique()}")

# Validação de que cada location está presente em ambos os conjuntos
for location in locations:
    train_count = len(train_df[train_df['Location'] == location])
    test_count = len(test_df[test_df['Location'] == location])
    total_count = train_count + test_count
    
    if train_count == 0 or test_count == 0:
        print(f"   {location}: treino={train_count}, teste={test_count} (PROBLEMA!)")
    
    if location == locations[0]:
        print(f"   {location}: treino={train_count} ({train_count/total_count*100:.1f}%), teste={test_count} ({test_count/total_count*100:.1f}%)")
    elif location == locations[-1]:
        print(f"   {location}: treino={train_count} ({train_count/total_count*100:.1f}%), teste={test_count} ({test_count/total_count*100:.1f}%)")

print(f"   Todas as {len(locations)} locations têm amostras em treino e teste!")

# Criação de amostra fixa para SHAP (extraída do conjunto de teste)
print(f"\nCriando amostra fixa para SHAP Analysis...")
print(f"   Amostras: {SHAP_SAMPLES} (do conjunto de teste)")

# Amostragem estratificada para SHAP
if len(test_idx) >= SHAP_SAMPLES:
    shap_idx = test_df.groupby('RainTomorrow', group_keys=False).apply(
        lambda x: x.sample(
            n=int(SHAP_SAMPLES * len(x) / len(test_df)),
            random_state=RANDOM_STATE
        )
    ).index
    
    if len(shap_idx) < SHAP_SAMPLES:
        remaining = SHAP_SAMPLES - len(shap_idx)
        remaining_idx = test_df[~test_df.index.isin(shap_idx)].sample(
            n=remaining, 
            random_state=RANDOM_STATE
        ).index
        shap_idx = shap_idx.union(remaining_idx)
    
    shap_idx = shap_idx[:SHAP_SAMPLES]
else:
    shap_idx = test_idx
    print(f"   Teste tem apenas {len(test_idx)} amostras. Usando todas.")

shap_df = df.loc[shap_idx]

print(f"   Amostras SHAP: {len(shap_idx)}")
print(f"   Distribuição SHAP:")
shap_class_dist = shap_df['RainTomorrow'].value_counts()
print(f"      Sem Chuva: {shap_class_dist[0]} ({shap_class_dist[0]/len(shap_idx)*100:.1f}%)")
print(f"      Chuva: {shap_class_dist[1]} ({shap_class_dist[1]/len(shap_idx)*100:.1f}%)")
print(f"   Locations SHAP: {shap_df['Location'].nunique()}")

# Salvamento dos índices em formato CSV
print(f"\nSalvando índices...")
train_indices_df = pd.DataFrame({
    'index': train_idx,
    'Location': df.loc[train_idx, 'Location'].values,
    'RainTomorrow': df.loc[train_idx, 'RainTomorrow'].values
})
train_indices_df.to_csv(train_indices_path, index=False)
print(f"   {train_indices_path.name}")

test_indices_df = pd.DataFrame({
    'index': test_idx,
    'Location': df.loc[test_idx, 'Location'].values,
    'RainTomorrow': df.loc[test_idx, 'RainTomorrow'].values
})
test_indices_df.to_csv(test_indices_path, index=False)
print(f"   {test_indices_path.name}")

shap_indices_df = pd.DataFrame({
    'index': shap_idx,
    'Location': df.loc[shap_idx, 'Location'].values,
    'RainTomorrow': df.loc[shap_idx, 'RainTomorrow'].values
})
shap_indices_df.to_csv(shap_indices_path, index=False)
print(f"   {shap_indices_path.name}")

print(f"\n" + "=" * 80)
print("DIVISÃO CRIADA COM SUCESSO!")
print("=" * 80)
print(f"\nResumo:")
print(f"   Total de amostras: {len(df):,}")
print(f"   Treino (80%): {len(train_idx):,} amostras")
print(f"   Teste (20%): {len(test_idx):,} amostras")
print(f"   SHAP (amostra fixa): {len(shap_idx):,} amostras")
print(f"   Locations: {len(locations)}")
print(f"   Random state: {RANDOM_STATE}")
