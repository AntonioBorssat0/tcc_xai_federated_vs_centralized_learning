"""
Cliente Flower para treinamento federado com MLP em PyTorch.
Implementa a lógica de treinamento local e avaliação para cada nó do sistema federado.
"""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from federated_mlp.task import (
    WeatherMLP, get_weights, load_data, set_weights, test, train
)

import logging

# Configuração de logging para evitar mensagens duplicadas
logging.getLogger('flwr').propagate = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Definição do cliente Flower para treinamento federado
class FlowerClient(NumPyClient):
    """
    Cliente Flower para treinamento federado.
    
    Gerencia o treinamento local do modelo em cada partição de dados,
    atualiza os pesos do modelo global e avalia o desempenho local.
    """
    def __init__(self, net, trainloader, valloader, local_epochs, learning_rate):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        """Executa treinamento local e retorna pesos atualizados."""
        logger.info(f"Iniciando treinamento - Rodada {config.get('server_round', 0)}")
        set_weights(self.net, parameters)
        train_loss = train(self.net, self.trainloader, self.local_epochs, self.device, self.learning_rate)
        logger.info(f"Treinamento concluído - Loss: {train_loss:.4f}")
        
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        """Avalia o modelo global no conjunto de validação local."""
        set_weights(self.net, parameters)
        loss, num_samples, metrics = test(self.net, self.valloader, self.device)
        return loss, num_samples, metrics


def client_fn(context: Context):
    """
    Função de factory para criar instâncias do cliente.
    
    Carrega os dados particionados, configura o modelo com os hiperparâmetros
    especificados e retorna uma instância do FlowerClient.
    
    Args:
        context: Contexto do Flower contendo configurações de nó e execução
    
    Returns:
        FlowerClient: Instância configurada do cliente
    """
    # Identificação da partição
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    # Extração de hiperparâmetros da configuração
    batch_size = context.run_config.get("batch-size", 32)
    test_split = context.run_config.get("test-split", 0.2)
    use_location = context.run_config.get("use-location-feature", False)
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config.get("learning-rate", 0.001)
    
    # Hiperparâmetros da arquitetura do modelo
    hidden1 = context.run_config.get("hidden-layer-1", 128)
    hidden2 = context.run_config.get("hidden-layer-2", 64)
    hidden3 = context.run_config.get("hidden-layer-3", 32)
    dropout1 = context.run_config.get("dropout-1", 0.3)
    dropout2 = context.run_config.get("dropout-2", 0.2)
    
    # Carregamento dos dados particionados
    trainloader, valloader, testloader, input_size = load_data(
        partition_id, num_partitions,
        batch_size=batch_size,
        test_split=test_split,
        use_location=use_location
    )

    # Inicialização do modelo com a arquitetura configurada
    net = WeatherMLP(input_size, hidden1, hidden2, hidden3, dropout1, dropout2)
    
    return FlowerClient(net, trainloader, valloader, local_epochs, learning_rate).to_client()


# Aplicação cliente do Flower
app = ClientApp(client_fn)