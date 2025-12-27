"""
Carregamento e preprocessamento de dados para predição de clima.
Suporta encoding cíclico de direções de vento e particionamento por localização.
"""

import math
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

# Configurações do dataset
LABEL_COL = "RainTomorrow"
LOCATION_COL = "Location"
WIND_COLS = ("WindGustDir", "WindDir9am", "WindDir3pm")

# 16 direções cardeais (em graus)
DIR_TO_DEG = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
    "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
}


def _encode_wind_directions_cyclic(df: pd.DataFrame, wind_cols=WIND_COLS):
    """
    Codifica direções de vento como componentes cíclicos sin/cos.
    
    Args:
        df: DataFrame com colunas de direção de vento
        wind_cols: Tupla com nomes das colunas de vento
    
    Returns:
        DataFrame com colunas _sin e _cos substituindo direções originais
    
    Nota: Valores ausentes/desconhecidos são substituídos por 0.0
    """
    df = df.copy()
    for col in wind_cols:
        if col not in df.columns:
            continue

        # Normalização de strings e mapeamento para graus
        raw = df[col].astype(str).str.strip().str.upper()
        mapped_deg = raw.map(DIR_TO_DEG).astype(float)

        rad = np.deg2rad(mapped_deg)
        sin_col = f"{col}_sin"
        cos_col = f"{col}_cos"

        # Cálculo de sin/cos; NaN substituído por 0.0
        df[sin_col] = np.sin(rad).fillna(0.0)
        df[cos_col] = np.cos(rad).fillna(0.0)

        df.drop(columns=[col], inplace=True)

    return df


def _preprocess_train_valid(train_df: pd.DataFrame, valid_df: pd.DataFrame, keep_location=False):
    """
    Preprocessa DataFrames de treino e validação e retorna xgb.DMatrix.
    
    Args:
        train_df: DataFrame de treinamento
        valid_df: DataFrame de validação
        keep_location: Se True, mantém Location como feature categórica (para comparação).
                      Se False, remove Location (padrão para federado).
    
    Returns:
        tuple: (dtrain, dvalid) como objetos xgb.DMatrix
    
    Etapas:
      - Aplica encoding cíclico para colunas de direção de vento
      - Trata coluna Location baseado no parâmetro keep_location
      - Garante que target seja numérico (0/1)
      - Imputa features numéricas com medianas do treino
      - Converte features não-numéricas restantes para numérico
    """
    train = train_df.copy()
    valid = valid_df.copy()

    # Conversão de label para numérico se necessário (trata 'Yes'/'No' ou 0/1)
    for df_name, df in [("train", train), ("valid", valid)]:
        if LABEL_COL in df.columns:
            if df[LABEL_COL].dtype == object:
                df[LABEL_COL] = df[LABEL_COL].map({"Yes": 1, "No": 0})
            df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0).astype(int)

        # Conversão de RainToday para numérico se necessário
        if 'RainToday' in df.columns:
            if df['RainToday'].dtype == object:
                df['RainToday'] = df['RainToday'].map({"Yes": 1, "No": 0})
            df['RainToday'] = pd.to_numeric(df['RainToday'], errors="coerce").fillna(0).astype(int)
        
        if df_name == "train":
            train = df
        else:
            valid = df

    # Aplicação de encoding cíclico para direções de vento
    train = _encode_wind_directions_cyclic(train)
    valid = _encode_wind_directions_cyclic(valid)

    # Tratamento da coluna Location baseado no parâmetro
    if not keep_location:
        # Drop coluna location
        for df in (train, valid):
            if LOCATION_COL in df.columns:
                df.drop(columns=[LOCATION_COL], inplace=True)
    else:
        # Encode Location as categorical for centralized comparison
        if LOCATION_COL in train.columns and LOCATION_COL in valid.columns:
            # Usa categorias do treino como referência
            train[LOCATION_COL] = pd.Categorical(train[LOCATION_COL])
            location_categories = train[LOCATION_COL].cat.categories
            
            # Aplica mesmas categorias ao valid (trata localizações não vistas)
            valid[LOCATION_COL] = pd.Categorical(
                valid[LOCATION_COL], 
                categories=location_categories
            )
            
            # Conversão para códigos
            train[LOCATION_COL] = train[LOCATION_COL].cat.codes
            valid[LOCATION_COL] = valid[LOCATION_COL].cat.codes
            
            # Trata -1 (categorias não vistas em valid) -> substitui com 0
            if (valid[LOCATION_COL] == -1).any():
                valid[LOCATION_COL] = valid[LOCATION_COL].replace(-1, 0)

    # Separação de features (X) e target (y)
    y_train = train[LABEL_COL].values if LABEL_COL in train.columns else None
    y_valid = valid[LABEL_COL].values if LABEL_COL in valid.columns else None

    X_train = train.drop(columns=[LABEL_COL], errors="ignore")
    X_valid = valid.drop(columns=[LABEL_COL], errors="ignore")

    # Conversão de colunas não-numéricas para numérico e imputação com mediana do treino
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    non_numeric = [c for c in X_train.columns if c not in numeric_cols]
    for c in non_numeric:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_valid[c] = pd.to_numeric(X_valid[c], errors="coerce")
    
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    # Garantia de que X_valid possui as mesmas colunas que X_train
    for c in numeric_cols:
        if c not in X_valid.columns:
            X_valid[c] = np.nan
    
    # Remove colunas extras em X_valid que não estão no treino
    extra_cols = [c for c in X_valid.columns if c not in numeric_cols]
    if extra_cols:
        X_valid.drop(columns=extra_cols, inplace=True)

    # Imputação de colunas numéricas com mediana do treino
    for c in numeric_cols:
        median = X_train[c].median()
        X_train[c] = X_train[c].fillna(median)
        X_valid[c] = X_valid[c].fillna(median)

    # Criação de objetos DMatrix para XGBoost
    feature_names = X_train.columns.tolist()
    dtrain = xgb.DMatrix(X_train.values, label=y_train, feature_names=feature_names) if y_train is not None else xgb.DMatrix(X_train.values, feature_names=feature_names)
    dvalid = xgb.DMatrix(X_valid.values, label=y_valid, feature_names=feature_names) if y_valid is not None else xgb.DMatrix(X_valid.values, feature_names=feature_names)

    return dtrain, dvalid


def load_data(partition_id, keep_location=False):
    """
    Carrega dados particionados por Location e retorna DMatrix de treino/validação.
    Usa divisão FIXA train/val/test para comparação justa com centralizados e PyTorch federado.
    
    Args:
        partition_id: ID do cliente (0-43 para 44 localizações)
        keep_location: Se True, mantém Location como feature (para experimentos de comparação)
                      Se False, remove Location (padrão para aprendizado federado)
    
    Returns:
        tuple: (train_dmatrix, valid_dmatrix, num_train, num_val)
    """
    # Construção do caminho para o dataset (relativo a este arquivo)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    csv_path = project_root / "datasets" / "rain_australia" / "weatherAUS_cleaned.csv"
    
    df = pd.read_csv(csv_path)
    
    if df.empty:
        raise ValueError(f"Dataset is empty: {csv_path}")
    
    # Carregamento de índices FIXOS train/val/test (para comparação com centralizados e PyTorch)
    train_indices_path = project_root / "datasets" / "train_indices.csv"
    val_indices_path = project_root / "datasets" / "val_indices.csv"
    test_indices_path = project_root / "datasets" / "test_indices.csv"
    
    use_fixed_split = False
    if train_indices_path.exists() and val_indices_path.exists() and test_indices_path.exists():
        train_indices_df = pd.read_csv(train_indices_path)
        val_indices_df = pd.read_csv(val_indices_path)
        test_indices_df = pd.read_csv(test_indices_path)
        
        train_global_indices = set(train_indices_df['index'].values)
        val_global_indices = set(val_indices_df['index'].values)
        test_global_indices = set(test_indices_df['index'].values)
        use_fixed_split = True
        print(f"   Usando divisão FIXA train/val/test (para comparação)")
    else:
        print(f"   AVISO: train/val_indices.csv não encontrado! Usando divisão aleatória.")
    
    # Obtenção de localizações únicas e atribuição para esta partição
    locations = sorted(df[LOCATION_COL].unique())
    
    if partition_id >= len(locations):
        raise ValueError(
            f"partition_id ({partition_id}) exceeds number of locations ({len(locations)})"
        )
    
    # Cada cliente recebe UMA localização (44 clientes no total)
    client_location = locations[partition_id]
    partition_data = df[df[LOCATION_COL] == client_location].copy()
    
    if partition_data.empty:
        raise ValueError(f"Partition {partition_id} (Location: {client_location}) has no data!")
    
    # Divisão usando índices FIXOS se disponíveis
    if use_fixed_split:
        # Obtém índices que pertencem a esta localização E estão nos conjuntos train/val
        # NOTA: Para treinamento FL, usamos train para treino e VAL para validação local
        # (conjunto test é reservado apenas para avaliação final)
        partition_indices = set(partition_data.index)
        
        train_mask = partition_data.index.isin(train_global_indices)
        val_mask = partition_data.index.isin(val_global_indices)
        
        train_df = partition_data[train_mask].copy()
        valid_df = partition_data[val_mask].copy()
        
        if train_df.empty or valid_df.empty:
            print(f"   Aviso: Location {client_location} sem dados de train ou val")
            # Fallback para divisão 90/10
            from sklearn.model_selection import train_test_split
            train_df, valid_df = train_test_split(
                partition_data, test_size=0.1, random_state=42, 
                stratify=partition_data[LABEL_COL]
            )
    else:
        # Fallback: divisão aleatória 90/10 (train/val, test é separado)
        from sklearn.model_selection import train_test_split
        train_df, valid_df = train_test_split(
            partition_data, test_size=0.1, random_state=42,
            stratify=partition_data[LABEL_COL]
        )
    
    # Preprocessamento e obtenção de objetos DMatrix
    train_dmatrix, valid_dmatrix = _preprocess_train_valid(train_df, valid_df, keep_location=keep_location)
    
    num_train = len(train_df)
    num_val = len(valid_df)
    
    return train_dmatrix, valid_dmatrix, num_train, num_val


def replace_keys(input_dict, match="-", target="_"):
    """
    Substitui recursivamente string match por target nas chaves do dicionário.
    
    Args:
        input_dict: Dicionário de entrada
        match: String a ser substituída
        target: String de substituição
    
    Returns:
        Novo dicionário com chaves modificadas
    """
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
