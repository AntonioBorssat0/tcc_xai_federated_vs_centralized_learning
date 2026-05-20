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
    
    Retorna:
        DataFrame com colunas _sin e _cos substituindo as direções originais

    Observação: valores ausentes ou desconhecidos são substituídos por 0.0
    """
    df = df.copy()
    for col in wind_cols:
        if col not in df.columns:
            continue

        # Normaliza strings e converte para graus
        raw = df[col].astype(str).str.strip().str.upper()
        mapped_deg = raw.map(DIR_TO_DEG).astype(float)

        rad = np.deg2rad(mapped_deg)
        sin_col = f"{col}_sin"
        cos_col = f"{col}_cos"

        # Calcula sin/cos; NaN é substituído por 0.0
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
    
    Retorna:
        tuple: (dtrain, dvalid) como objetos xgb.DMatrix
    
    Etapas:
    - Aplica encoding cíclico nas colunas de direção de vento
      - Trata coluna Location baseado no parâmetro keep_location
      - Garante que target seja numérico (0/1)
      - Imputa features numéricas com medianas do treino
      - Converte features não-numéricas restantes para numérico
    """
    train = train_df.copy()
    valid = valid_df.copy()

    # Converte o rótulo para numérico, se necessário (trata 'Yes'/'No' ou 0/1)
    for df_name, df in [("train", train), ("valid", valid)]:
        if LABEL_COL in df.columns:
            if df[LABEL_COL].dtype == object:
                df[LABEL_COL] = df[LABEL_COL].map({"Yes": 1, "No": 0})
            df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0).astype(int)

        # Converte RainToday para numérico, se necessário
        if 'RainToday' in df.columns:
            if df['RainToday'].dtype == object:
                df['RainToday'] = df['RainToday'].map({"Yes": 1, "No": 0})
            df['RainToday'] = pd.to_numeric(df['RainToday'], errors="coerce").fillna(0).astype(int)
        
        if df_name == "train":
            train = df
        else:
            valid = df

    # Aplica o encoding cíclico nas direções de vento
    train = _encode_wind_directions_cyclic(train)
    valid = _encode_wind_directions_cyclic(valid)

    # Tratamento da coluna Location baseado no parâmetro
    if not keep_location:
        # Remove a coluna Location
        for df in (train, valid):
            if LOCATION_COL in df.columns:
                df.drop(columns=[LOCATION_COL], inplace=True)
    else:
        # Codifica Location como categórica para comparação com o modelo centralizado
        if LOCATION_COL in train.columns and LOCATION_COL in valid.columns:
            # Usa as categorias do treino como referência
            train[LOCATION_COL] = pd.Categorical(train[LOCATION_COL])
            location_categories = train[LOCATION_COL].cat.categories
            
            # Aplica as mesmas categorias ao conjunto de validação
            valid[LOCATION_COL] = pd.Categorical(
                valid[LOCATION_COL], 
                categories=location_categories
            )
            
            # Converte para códigos numéricos
            train[LOCATION_COL] = train[LOCATION_COL].cat.codes
            valid[LOCATION_COL] = valid[LOCATION_COL].cat.codes
            
            # Trata -1 (categorias não vistas na validação) substituindo por 0
            if (valid[LOCATION_COL] == -1).any():
                valid[LOCATION_COL] = valid[LOCATION_COL].replace(-1, 0)

    # Separa features (X) e alvo (y)
    y_train = train[LABEL_COL].values if LABEL_COL in train.columns else None
    y_valid = valid[LABEL_COL].values if LABEL_COL in valid.columns else None

    X_train = train.drop(columns=[LABEL_COL], errors="ignore")
    X_valid = valid.drop(columns=[LABEL_COL], errors="ignore")

    # Converte colunas não numéricas para numérico e imputa com a mediana do treino
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    non_numeric = [c for c in X_train.columns if c not in numeric_cols]
    for c in non_numeric:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_valid[c] = pd.to_numeric(X_valid[c], errors="coerce")
    
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    # Garante que X_valid tenha as mesmas colunas de X_train
    for c in numeric_cols:
        if c not in X_valid.columns:
            X_valid[c] = np.nan
    
    # Remove colunas extras de X_valid que não existem no treino
    extra_cols = [c for c in X_valid.columns if c not in numeric_cols]
    if extra_cols:
        X_valid.drop(columns=extra_cols, inplace=True)

    # Imputa colunas numéricas com a mediana do treino
    for c in numeric_cols:
        median = X_train[c].median()
        X_train[c] = X_train[c].fillna(median)
        X_valid[c] = X_valid[c].fillna(median)

    # Cria os objetos DMatrix para o XGBoost
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
    
    Retorna:
        tuple: (train_dmatrix, valid_dmatrix, num_train, num_val)
    """
    # Caminho relativo esperado para o dataset
    rel_dataset = Path("datasets") / "rain_australia" / "weatherAUS_cleaned.csv"

    # Locais candidatos para busca, na ordem:
    # 1) layout do repositório relativo a este arquivo
    # 2) diretório de trabalho atual
    # 3) pasta datasets do app empacotado pelo Flower
    candidates = []
    script_dir = Path(__file__).parent
    candidates.append(script_dir.parent.parent / rel_dataset)  # repo root style
    candidates.append(Path.cwd() / rel_dataset)  # running from workspace root
    candidates.append(Path.home() / ".flwr" / "apps" / "datasets" / "rain_australia" / "weatherAUS_cleaned.csv")

    # Também sobe alguns níveis a partir deste arquivo para ser robusto quando
    # o app estiver empacotado
    for i, anc in enumerate(script_dir.parents):
        candidates.append(anc / rel_dataset)
        if i >= 6:
            break

    # Sobe a partir do diretório atual para encontrar a raiz do repositório
    cwd = Path.cwd().resolve()
    candidates.append(cwd / rel_dataset)
    for i, anc in enumerate(cwd.parents):
        candidates.append(anc / rel_dataset)
        if i >= 6:
            break

    # Remove duplicatas preservando a ordem
    seen = set()
    candidates_filtered = []
    for p in candidates:
        p = p.resolve() if p.exists() else p
        if str(p) not in seen:
            seen.add(str(p))
            candidates_filtered.append(p)

    csv_path = None
    tried = []
    for cand in candidates_filtered:
        tried.append(str(cand))
        if cand.exists():
            csv_path = cand
            break

    if csv_path is None:
        raise FileNotFoundError(
            "Could not find weatherAUS_cleaned.csv. Tried these locations:\n  " +
            "\n  ".join(tried)
        )

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"Dataset is empty: {csv_path}")

    # Define project_root relativo ao local do dataset para que a lógica
    # seguinte (índices de treino/validação/teste) funcione tanto no repositório
    # quanto no app empacotado.
    # Se csv_path for .../repo/datasets/rain_australia/weatherAUS_cleaned.csv,
    # então csv_path.parents[2] == repo (raiz correta do projeto).
    try:
        if len(csv_path.parents) >= 3:
            project_root = csv_path.parents[2]
        else:
            project_root = csv_path.parent.parent
    except Exception:
        project_root = csv_path.parent.parent
    
    # Carrega os índices FIXOS de treino/validação/teste (para comparação com
    # os modelos centralizados e com o PyTorch federado)
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
    
    # Obtém as localizações únicas e atribui a partição atual
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
    
    # Divide usando índices FIXOS, se disponíveis
    if use_fixed_split:
        # Obtém índices que pertencem a esta localização e estão nos conjuntos
        # train/val. Para o treinamento federado, usamos train para treino e val
        # para validação local; test fica reservado para a avaliação final.
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
        # Fallback: divisão aleatória 90/10 (train/val, test fica separado)
        from sklearn.model_selection import train_test_split
        train_df, valid_df = train_test_split(
            partition_data, test_size=0.1, random_state=42,
            stratify=partition_data[LABEL_COL]
        )
    
    # Preprocessa e cria os objetos DMatrix
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
    
    Retorna:
        Novo dicionário com as chaves modificadas
    """
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
