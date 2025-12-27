"""
Módulo de tarefas para treinamento federado com PyTorch.
Implementa o modelo MLP, processamento de dados e funções de treinamento/avaliação.
"""

from collections import OrderedDict
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score



# Constantes do dataset de clima
LABEL_COL = "RainTomorrow"
LOCATION_COL = "Location"
WIND_DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", 
             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

class WeatherMLP(nn.Module):
    """
    Modelo MLP para predição de clima.
    
    Arquitetura: 3 camadas ocultas com ReLU e Dropout.
    Saída: 1 neurônio (classificação binária - choverá amanhã?).
    """
    
    def __init__(self, input_size, hidden1=128, hidden2=64, hidden3=32, dropout1=0.3, dropout2=0.2):
        super(WeatherMLP, self).__init__()
        # Armazenamento dos parâmetros de arquitetura (usado para reconstrução do modelo)
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, 1)
        )
        
    def forward(self, x):
        """Propagação forward. Retorna logits (sem sigmoid)."""
        return self.layers(x).squeeze(-1)

# Funções auxiliares de preprocessamento
def encode_wind_directions(df):
    """Converte direções de vento categóricas em componentes seno e cosseno (encoding cíclico)."""
    wind_cols = ["WindGustDir", "WindDir9am", "WindDir3pm"]
    
    for col in wind_cols:
        if col in df.columns:
            # Criação de dicionário mapeando direções para ângulos (em radianos)
            dir_to_rad = {dir_: i * 2 * np.pi / 16 for i, dir_ in enumerate(WIND_DIRS)}
            
            # Tratamento de valores ausentes
            dir_to_rad.update({np.nan: 0, "NA": 0, None: 0})
            
            # Conversão para ângulos e depois para componentes sin/cos
            angles = df[col].map(dir_to_rad)
            df[f"{col}_sin"] = np.sin(angles)
            df[f"{col}_cos"] = np.cos(angles)
            df.drop(columns=[col], inplace=True)
            
    return df

def prepare_weather_data(df, use_location=False):
    """
    Processa o dataframe de clima para machine learning.
    
    Args:
        df: DataFrame com dados de clima
        use_location: Se True, faz one-hot encoding de Location; se False, remove
    
    Returns:
        DataFrame processado
    """
    # Dataset já está limpo, RainToday e RainTomorrow já são 0/1
    # Apenas precisa codificar direções de vento
    
    # Tratamento de direções de vento (converte categórico para sin/cos)
    df = encode_wind_directions(df)
    
    # Tratamento da coluna Location
    if use_location:
        # One-hot encoding de Location (se usuário quiser incluir como feature)
        df = pd.get_dummies(df, columns=[LOCATION_COL], prefix='Location', drop_first=True)
    else:
        # Remove coluna location (padrão - já tratado pelo particionamento federado)
        if LOCATION_COL in df.columns:
            df = df.drop(columns=[LOCATION_COL])
    
    # Remove coluna Date se presente
    if 'Date' in df.columns:
        df = df.drop(columns=['Date'])
    
    return df

def load_data(partition_id, num_partitions, 
              batch_size=32, test_split=0.2, use_location=False,
              data_path=None):
    """
    Carrega dados de clima particionados usando divisão FIXA de train/val/test.
    
    Args:
        partition_id: ID da partição (corresponde a locações geográficas)
        num_partitions: Número total de partições
        batch_size: Tamanho do batch para DataLoaders
        test_split: Fração de teste (usado apenas se split fixo não existir)
        use_location: Se True, inclui Location como feature
        data_path: Caminho para o arquivo de dados (None usa padrão)
    
    Returns:
        tuple: (trainloader, valloader, testloader, input_size)
    """
    # Default data path (relative to this file)
    from pathlib import Path
    if data_path is None:
        script_dir = Path(__file__).parent  # flwr_pytorch_test/
        project_root = script_dir.parent.parent
        data_path = str(project_root / "datasets" / "rain_australia" / "weatherAUS_cleaned.csv")
    
    # Load the Australian weather dataset
    df = pd.read_csv(data_path)
    
    # Validate dataset
    if df.empty:
        raise ValueError(f"Dataset is empty: {data_path}")
    
    # Carregamento de índices fixos de train/val/test (para comparação com centralizado)
    from pathlib import Path
    data_dir = Path(data_path).parent.parent
    train_indices_path = data_dir / "train_indices.csv"
    val_indices_path = data_dir / "val_indices.csv"
    test_indices_path = data_dir / "test_indices.csv"
    
    use_fixed_split = False
    if train_indices_path.exists() and val_indices_path.exists() and test_indices_path.exists():
        print(f"   Usando divisão FIXA train/val/test (para comparação com centralizado)")
        train_indices_df = pd.read_csv(train_indices_path)
        val_indices_df = pd.read_csv(val_indices_path)
        test_indices_df = pd.read_csv(test_indices_path)
        
        train_global_indices = set(train_indices_df['index'].values)
        val_global_indices = set(val_indices_df['index'].values)
        test_global_indices = set(test_indices_df['index'].values)
        use_fixed_split = True
    else:
        print(f"   AVISO: train/val_indices.csv não encontrado! Usando split aleatório.")
    
    # Criação de partições baseadas em localização
    locations = df[LOCATION_COL].unique()
    
    # Validate partition parameters
    if partition_id >= len(locations):
        raise ValueError(
            f"partition_id ({partition_id}) exceeds number of locations ({len(locations)})"
        )
    
    if num_partitions > len(locations):
        print(f"Aviso: num_partitions ({num_partitions}) > locations ({len(locations)}). "
              f"Alguns clientes terão múltiplas locações.")
    
    location_partitions = np.array_split(locations, num_partitions)
    
    # Obtenção das locações para esta partição
    partition_locations = location_partitions[partition_id]
    partition_data = df[df[LOCATION_COL].isin(partition_locations)].copy()
    
    # Validação de que a partição possui dados
    if partition_data.empty:
        raise ValueError(f"Partition {partition_id} has no data!")
    
    # Processamento dos dados ANTES de dividir (importante para preservação de índices)
    processed_df = prepare_weather_data(partition_data, use_location=use_location)
    
    # Separação de features e target
    X = processed_df.drop(columns=[LABEL_COL])
    y = processed_df[LABEL_COL]
    
    # Divisão usando índices FIXOS se disponíveis
    if use_fixed_split:
        # Filtro por índices globais de train/val/test
        # Nota: processed_df preserva índices originais de df
        train_mask = processed_df.index.isin(train_global_indices)
        val_mask = processed_df.index.isin(val_global_indices)
        test_mask = processed_df.index.isin(test_global_indices)
        
        X_train = X[train_mask]
        X_val = X[val_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_val = y[val_mask]
        y_test = y[test_mask]
        
        print(f"Partição {partition_id}: Locations = {list(partition_locations)}, "
              f"Train = {len(X_train)}, Val = {len(X_val)}, Test = {len(X_test)}")
    else:
        # Fallback: split aleatório
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        print(f"Partição {partition_id}: Locations = {list(partition_locations)}, "
              f"Samples = {len(partition_data)} (split aleatório)")
    
    # Padronização de features (ajuste apenas no conjunto de treino)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Tratamento de valores NaN do StandardScaler (quando variância = 0)
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Conversão para tensores PyTorch
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train.values)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val.values)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test.values)
    
    # Criação de TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Criação de DataLoaders
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size)
    testloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return trainloader, valloader, testloader, X_train.shape[1]


def train(net, trainloader, epochs, device, learning_rate=0.001):
    """
    Treina o modelo no conjunto de treinamento.
    
    Args:
        net: Modelo a ser treinado
        trainloader: DataLoader com dados de treino
        epochs: Número de épocas
        device: Dispositivo (CPU/GPU)
        learning_rate: Taxa de aprendizado
    
    Returns:
        float: Perda média de treinamento
    """
    net.to(device)
    
    # Cálculo de pesos de classe para dados desbalanceados
    all_labels = []
    for _, labels in trainloader:
        all_labels.extend(labels.numpy())
    
    pos_count = sum(all_labels)
    neg_count = len(all_labels) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count]).to(device)
    
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net.train()
    running_loss = 0.0
    
    for _ in range(epochs):
        for data, targets in trainloader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = net(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss

def test(net, testloader, device):
    """
    Valida o modelo no conjunto de teste.
    
    Args:
        net: Modelo a ser avaliado
        testloader: DataLoader com dados de teste
        device: Dispositivo (CPU/GPU)
    
    Returns:
        tuple: (perda, num_amostras, métricas)
    """    
    net.to(device)
    net.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    
    all_preds = []
    all_targets = []
    loss = 0.0
    
    with torch.no_grad():
        for data, targets in testloader:
            data, targets = data.to(device), targets.to(device)
            outputs = net(data)
            loss += criterion(outputs, targets).item()
            
            # Aplica sigmoid para cálculo de métricas
            outputs_prob = torch.sigmoid(outputs)
            all_preds.extend(outputs_prob.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Conversão para predições binárias
    binary_preds = (np.array(all_preds) > 0.5).astype(int)
    all_targets = np.array(all_targets)
    
    # Cálculo de métricas
    accuracy = (binary_preds == all_targets).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, binary_preds, average='binary'
    )
    auc = roc_auc_score(all_targets, all_preds)
    
    from sklearn.metrics import average_precision_score
    avg_precision = average_precision_score(all_targets, all_preds)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "avg_precision": avg_precision
    }
    
    return loss / len(testloader), len(testloader.dataset), metrics

def get_weights(net):
    """Extrai pesos do modelo como lista de arrays NumPy."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    """Define pesos do modelo a partir de lista de arrays NumPy."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)