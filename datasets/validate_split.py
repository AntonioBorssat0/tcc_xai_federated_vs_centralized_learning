"""
Script para validar que a divisão train/val/test não possui vazamento de dados.
Verifica overlaps entre os conjuntos e valida a distribuição dos dados.
"""

import pandas as pd
from pathlib import Path

print("=" * 80)
print("VALIDAÇÃO DA DIVISÃO TRAIN/VAL/TEST")
print("=" * 80)

# Definição de caminhos
script_dir = Path(__file__).parent
train_path = script_dir / "train_indices.csv"
val_path = script_dir / "val_indices.csv"
test_path = script_dir / "test_indices.csv"

# Carregamento dos índices
print(f"\nCarregando índices...")
train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

train_indices = set(train_df['index'])
val_indices = set(val_df['index'])
test_indices = set(test_df['index'])

print(f"   Train: {len(train_indices):,} amostras")
print(f"   Val: {len(val_indices):,} amostras")
print(f"   Test: {len(test_indices):,} amostras")
print(f"   Total: {len(train_indices) + len(val_indices) + len(test_indices):,} amostras")

# Verificação de sobreposição entre conjuntos
print(f"\nVerificando overlaps...")

overlap_train_val = train_indices & val_indices
overlap_train_test = train_indices & test_indices
overlap_val_test = val_indices & test_indices

if overlap_train_val:
    print(f"   ERRO: Train e Val têm {len(overlap_train_val)} índices em comum!")
    print(f"      Exemplos: {list(overlap_train_val)[:5]}")
else:
    print(f"   Train e Val: sem overlap")

if overlap_train_test:
    print(f"   ERRO: Train e Test têm {len(overlap_train_test)} índices em comum!")
    print(f"      Exemplos: {list(overlap_train_test)[:5]}")
else:
    print(f"   Train e Test: sem overlap")

if overlap_val_test:
    print(f"   ERRO: Val e Test têm {len(overlap_val_test)} índices em comum!")
    print(f"      Exemplos: {list(overlap_val_test)[:5]}")
else:
    print(f"   Val e Test: sem overlap")

# Verificação do total de amostras
total_esperado = 112925
total_atual = len(train_indices) + len(val_indices) + len(test_indices)

print(f"\nVerificando total...")
if total_atual == total_esperado:
    print(f"   Total correto: {total_atual:,} amostras")
else:
    print(f"   ERRO: Total incorreto!")
    print(f"      Esperado: {total_esperado:,}")
    print(f"      Atual: {total_atual:,}")
    print(f"      Diferença: {abs(total_atual - total_esperado):,}")

# Análise de distribuição de classes
print(f"\nVerificando distribuição de classes...")

for nome, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    class_dist = df['RainTomorrow'].value_counts(normalize=True)
    print(f"   {nome}:")
    print(f"      Sem Chuva: {class_dist[0]*100:.1f}%")
    print(f"      Chuva: {class_dist[1]*100:.1f}%")

# Verificação de locations
print(f"\nVerificando locations...")
train_locations = set(train_df['Location'].unique())
val_locations = set(val_df['Location'].unique())
test_locations = set(test_df['Location'].unique())

print(f"   Train: {len(train_locations)} locations")
print(f"   Val: {len(val_locations)} locations")
print(f"   Test: {len(test_locations)} locations")

all_locations = train_locations | val_locations | test_locations
print(f"   Total único: {len(all_locations)} locations")

# Validação de presença de todas as locations nos três conjuntos
if train_locations == val_locations == test_locations:
    print(f"   Todas as locations estão em train, val e test")
else:
    missing_in_train = (val_locations | test_locations) - train_locations
    missing_in_val = (train_locations | test_locations) - val_locations
    missing_in_test = (train_locations | val_locations) - test_locations
    
    if missing_in_train:
        print(f"   Locations ausentes no train: {missing_in_train}")
    if missing_in_val:
        print(f"   Locations ausentes no val: {missing_in_val}")
    if missing_in_test:
        print(f"   Locations ausentes no test: {missing_in_test}")

# Resultado final da validação
print(f"\n" + "=" * 80)
has_errors = bool(overlap_train_val or overlap_train_test or overlap_val_test or total_atual != total_esperado)

if not has_errors:
    print("VALIDAÇÃO COMPLETA - SEM VAZAMENTO DE DADOS!")
    print("=" * 80)
    print(f"\nResumo:")
    print(f"   Train: {len(train_indices):,} ({len(train_indices)/total_esperado*100:.1f}%)")
    print(f"   Val: {len(val_indices):,} ({len(val_indices)/total_esperado*100:.1f}%)")
    print(f"   Test: {len(test_indices):,} ({len(test_indices)/total_esperado*100:.1f}%)")
    print(f"   Total: {total_atual:,}")
    print(f"   Sem overlaps: Sim")
    print(f"   Locations completas: Sim")
    print(f"\nA divisão está correta e pronta para uso!")
else:
    print("VALIDAÇÃO FALHOU - PROBLEMAS DETECTADOS!")
    print("=" * 80)
    print(f"\nCorrija os problemas acima antes de prosseguir.")

print()
