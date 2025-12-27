"""
Servidor Flower para treinamento federado com MLP em PyTorch.
Implementa a estratégia de agregação e gerenciamento do modelo global.
"""

import logging

# Configuração de logging para evitar mensagens duplicadas
logging.getLogger('flwr').propagate = False

from typing import List, Tuple, Dict, Optional
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from federated_mlp.task import WeatherMLP, get_weights, set_weights
import torch
from pathlib import Path
import joblib
from datetime import datetime


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Agrega métricas de múltiplos clientes usando média ponderada.
    
    Args:
        metrics: Lista de tuplas (num_exemplos, dict_métricas) de cada cliente
        
    Returns:
        Dicionário de métricas agregadas
    """
    # Inicialização dos agregadores
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    # Obtém todas as chaves de métrica do primeiro cliente
    if not metrics:
        return {}
    
    metric_keys = metrics[0][1].keys()
    aggregated = {}
    
    # Calcula média ponderada para cada métrica
    for key in metric_keys:
        weighted_sum = sum(
            num_examples * m[key] 
            for num_examples, m in metrics 
            if key in m
        )
        aggregated[key] = weighted_sum / total_examples
    
    return aggregated


def server_fn(context: Context):
    """Cria a configuração do servidor federado."""
    # Leitura das configurações
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    
    default_input_size = 19
    input_size = context.run_config.get("input-size", default_input_size)
    
    # Configurações adicionais da estratégia
    fraction_evaluate = context.run_config.get("fraction-evaluate", 1.0)
    min_available_clients = context.run_config.get("min-available-clients", 2)
    
    # Hiperparâmetros da arquitetura do modelo
    hidden1 = context.run_config.get("hidden-layer-1", 128)
    hidden2 = context.run_config.get("hidden-layer-2", 64)
    hidden3 = context.run_config.get("hidden-layer-3", 32)
    dropout1 = context.run_config.get("dropout-1", 0.3)
    dropout2 = context.run_config.get("dropout-2", 0.2)
    
    # Inicialização dos parâmetros do modelo com a arquitetura especificada
    ndarrays = get_weights(WeatherMLP(
        input_size=default_input_size,
        hidden1=hidden1,
        hidden2=hidden2,
        hidden3=hidden3,
        dropout1=dropout1,
        dropout2=dropout2
    ))
    parameters = ndarrays_to_parameters(ndarrays)
    
    # Criação do modelo global para salvamento posterior
    global_model = WeatherMLP(
        input_size=default_input_size,
        hidden1=hidden1,
        hidden2=hidden2,
        hidden3=hidden3,
        dropout1=dropout1,
        dropout2=dropout2
    )
    
    # Definição da estratégia com agregação de métricas e salvamento do modelo
    class FedAvgWithSave(FedAvg):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Rastreamento do histórico de treinamento
            self.history = {
                'round': [],
                'train_loss': [],
                'train_accuracy': [],
                'val_loss': [],
                'val_accuracy': [],
                'val_auc': [],
                'val_aucpr': []
            }
        
        def aggregate_fit(self, server_round, results, failures):
            """Agrega pesos do modelo e salva o modelo final."""
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(
                server_round, results, failures
            )
            
            # Salvamento de métricas de treino no histórico
            if aggregated_metrics:
                # Verifica se esta rodada já existe (evaluate aconteceu primeiro)
                if server_round in self.history['round']:
                    # Atualiza entrada existente
                    idx = self.history['round'].index(server_round)
                    self.history['train_loss'][idx] = aggregated_metrics.get('loss', None)
                    self.history['train_accuracy'][idx] = aggregated_metrics.get('accuracy', None)
                else:
                    # Adiciona nova entrada (fit acontece antes de evaluate)
                    self.history['round'].append(server_round)
                    self.history['train_loss'].append(aggregated_metrics.get('loss', None))
                    self.history['train_accuracy'].append(aggregated_metrics.get('accuracy', None))
                    # Adiciona placeholders para métricas de validação
                    self.history['val_loss'].append(None)
                    self.history['val_accuracy'].append(None)
                    self.history['val_auc'].append(None)
                    self.history['val_aucpr'].append(None)
            
            # Salvamento de checkpoint do modelo a cada rodada (para análise de early stopping)
            if aggregated_parameters is not None:
                from flwr.common import parameters_to_ndarrays
                params = parameters_to_ndarrays(aggregated_parameters)
                set_weights(global_model, params)
                
                # Criação do diretório de checkpoints
                save_dir = Path("models")
                save_dir.mkdir(exist_ok=True)
                checkpoints_dir = save_dir / "checkpoints"
                checkpoints_dir.mkdir(exist_ok=True)
                
                # Salvamento do checkpoint para esta rodada
                checkpoint_path = checkpoints_dir / f"round_{server_round:03d}.pt"
                torch.save({
                    'model_state_dict': global_model.state_dict(),
                    'round': server_round,
                    'input_size': default_input_size,
                    'hidden1': hidden1,
                    'hidden2': hidden2,
                    'hidden3': hidden3,
                    'dropout1': dropout1,
                    'dropout2': dropout2,
                }, checkpoint_path)
            
            # Salvamento do modelo final após a última rodada
            if server_round == num_rounds and aggregated_parameters is not None:
                print(f"Salvando modelo global após rodada {server_round}...")
                
                # Conversão dos parâmetros de volta para pesos do modelo
                from flwr.common import parameters_to_ndarrays
                params = parameters_to_ndarrays(aggregated_parameters)
                set_weights(global_model, params)
                
                # Criação do diretório models se não existir
                save_dir = Path("models")
                save_dir.mkdir(exist_ok=True)
                
                # Salvamento do checkpoint legado Torch (compatibilidade retroativa)
                legacy_path = save_dir / "global_model_final.pt"
                torch.save({
                    'model_state_dict': global_model.state_dict(),
                    'input_size': default_input_size,
                    'hidden1': hidden1,
                    'hidden2': hidden2,
                    'hidden3': hidden3,
                    'dropout1': dropout1,
                    'dropout2': dropout2,
                    'num_rounds': num_rounds,
                }, legacy_path)
                print(f"Checkpoint legado salvo: {legacy_path.absolute()}")

                # Salvamento do artefato JOBLIB padronizado
                joblib_path = save_dir / "global_model_final.joblib"
                model_data = {
                    'best_model': global_model,
                    'best_params': None,
                    'best_score': None,
                    'study': None,
                    'scaler': None,
                    'feature_names': None,
                    'train_auc': None,
                    'test_auc': None,
                    'metadata': {
                        'saved_at': datetime.now().isoformat(timespec='seconds'),
                        'model_type': 'MLP',
                        'framework': 'Flower Federated',
                        'strategy': 'FedAvg',
                        'num_rounds': num_rounds,
                        'input_size': default_input_size,
                        'hidden_layers': [hidden1, hidden2, hidden3],
                        'dropouts': [dropout1, dropout2],
                    }
                }
                joblib.dump(model_data, joblib_path)
                print(f"JOBLIB salvo: {joblib_path.absolute()}")
                
                # Salvamento do histórico de treinamento
                print("Salvando histórico de treinamento...")
                import pandas as pd
                history_df = pd.DataFrame(self.history)
                
                history_csv_path = save_dir / "training_history.csv"
                history_df.to_csv(history_csv_path, index=False)
                print(f"History CSV salvo: {history_csv_path.absolute()}")
                
                history_json_path = save_dir / "training_history.json"
                history_df.to_json(history_json_path, orient='records', indent=2)
                print(f"History JSON salvo: {history_json_path.absolute()}")
                print(f"   ({len(self.history['round'])} rodadas)")
                
                # Análise de early stopping
                print("\n" + "="*80)
                print("ANÁLISE DE EARLY STOPPING")
                print("="*80)
                
                # Busca pela melhor rodada baseada em val_aucpr
                val_aucpr = [x if x is not None else 0.0 for x in self.history['val_aucpr']]
                if val_aucpr:
                    best_idx = val_aucpr.index(max(val_aucpr))
                    best_round = self.history['round'][best_idx]
                    best_aucpr = self.history['val_aucpr'][best_idx]
                    best_auc = self.history['val_auc'][best_idx]
                    best_acc = self.history['val_accuracy'][best_idx]
                    
                    print(f"\nMELHOR RODADA: {best_round}/{num_rounds}")
                    print(f"   Val AUCPR: {best_aucpr:.6f}")
                    print(f"   Val AUC:   {best_auc:.6f}")
                    print(f"   Val Acc:   {best_acc:.6f}")
                    
                    rounds_overfit = num_rounds - best_round
                    if rounds_overfit > 0:
                        pct_overfit = (rounds_overfit / num_rounds) * 100
                        print(f"\nOVERFITTING DETECTADO:")
                        print(f"   Rodadas após a melhor: {rounds_overfit}/{num_rounds} ({pct_overfit:.1f}%)")
                        print(f"   Recomendação: Poderia ter parado na rodada {best_round}")
                        print(f"   Checkpoint a usar: models/checkpoints/round_{best_round:03d}.pt")
                    else:
                        print(f"\nTreinamento terminou na rodada ótima")
                    
                    # Exibição de degradação de desempenho se houver
                    final_aucpr = self.history['val_aucpr'][-1]
                    if final_aucpr is not None and final_aucpr < best_aucpr:
                        aucpr_drop = best_aucpr - final_aucpr
                        pct_drop = (aucpr_drop / best_aucpr) * 100
                        print(f"\nQUEDA DE DESEMPENHO:")
                        print(f"   Do melhor ao final: {aucpr_drop:.6f} ({pct_drop:.2f}%)")
                
                print("="*80 + "\n")
            
            return aggregated_parameters, aggregated_metrics
        
        def aggregate_evaluate(self, server_round, results, failures):
            """Agrega resultados de avaliação e salva no histórico."""
            aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
                server_round, results, failures
            )
            
            # Salvamento de métricas de avaliação no histórico
            if aggregated_metrics:
                # Verifica se esta rodada já existe no histórico
                if server_round in self.history['round']:
                    # Atualiza entrada existente (fit aconteceu primeiro)
                    idx = self.history['round'].index(server_round)
                    self.history['val_loss'][idx] = aggregated_loss
                    self.history['val_accuracy'][idx] = aggregated_metrics.get('accuracy', None)
                    self.history['val_auc'][idx] = aggregated_metrics.get('auc', None)
                    self.history['val_aucpr'][idx] = aggregated_metrics.get('avg_precision', None)
                else:
                    # Rodada ainda não está no histórico (evaluate acontece antes de fit)
                    # Adiciona entrada placeholder que será atualizada por aggregate_fit
                    self.history['round'].append(server_round)
                    self.history['train_loss'].append(None)
                    self.history['train_accuracy'].append(None)
                    self.history['val_loss'].append(aggregated_loss)
                    self.history['val_accuracy'].append(aggregated_metrics.get('accuracy', None))
                    self.history['val_auc'].append(aggregated_metrics.get('auc', None))
                    self.history['val_aucpr'].append(aggregated_metrics.get('avg_precision', None))
            
            return aggregated_loss, aggregated_metrics
    
    strategy = FedAvgWithSave(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=min_available_clients,
        initial_parameters=parameters,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Criação da aplicação servidor do Flower
app = ServerApp(server_fn=server_fn)