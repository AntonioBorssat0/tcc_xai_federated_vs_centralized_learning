"""
Servidor Flower para treinamento federado com XGBoost.
Suporta estratégias Bagging e Cyclic.
"""

import numpy as np
import xgboost as xgb
import pandas as pd
from pathlib import Path
import joblib
from datetime import datetime
from flwr.app import ArrayRecord, Context
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedXgbBagging, FedXgbCyclic

from federated_xgboost.task import replace_keys

app = ServerApp()

# Variável global para armazenar histórico de treinamento
training_history = []


def log_evaluation_metrics(server_round, replies):
    """
    Registra métricas de avaliação dos clientes.
    
    Args:
        server_round: Rodada atual do servidor
        replies: Lista de mensagens de resposta dos clientes
    """
    if not replies:
        return
    
    # Extração de AUCPR de todas as mensagens dos clientes
    aucpr_values = []
    for msg in replies:
        if hasattr(msg, 'content') and 'metrics' in msg.content:
            metrics = msg.content['metrics']
            if hasattr(metrics, '__getitem__') and "aucpr" in metrics:
                aucpr_values.append(metrics["aucpr"])
    
    if aucpr_values:
        aucpr_mean = np.mean(aucpr_values)
        aucpr_std = np.std(aucpr_values)
        
        training_history.append({
            'round': server_round,
            'aucpr_mean': aucpr_mean,
            'aucpr_std': aucpr_std,
            'num_clients_evaluated': len(aucpr_values)
        })
        
        print(f"   Round {server_round}: AUCPR = {aucpr_mean:.4f} ± {aucpr_std:.4f} ({len(aucpr_values)} clientes)")


class FedXgbBaggingWithHistory(FedXgbBagging):
    """Estratégia FedXgbBagging com registro de histórico de treinamento."""
    
    def aggregate_evaluate(self, server_round, replies):
        """Agrega avaliações e registra métricas."""
        log_evaluation_metrics(server_round, replies)
        return super().aggregate_evaluate(server_round, replies)


class FedXgbCyclicWithHistory(FedXgbCyclic):
    """Estratégia FedXgbCyclic com registro de histórico de treinamento."""
    
    def aggregate_evaluate(self, server_round, replies):
        """Agrega avaliações e registra métricas."""
        log_evaluation_metrics(server_round, replies)
        return super().aggregate_evaluate(server_round, replies)


def save_training_history(history, strategy_name, models_dir):
    """
    Salva histórico de treinamento em CSV.
    
    Args:
        history: Lista de dicionários com histórico de métricas
        strategy_name: Nome da estratégia (bagging ou cyclic)
        models_dir: Diretório para salvar o arquivo
    """
    if not history:
        print("   AVISO: Nenhum histórico de treinamento para salvar")
        return
    
    df = pd.DataFrame(history)
    csv_path = models_dir / f"training_history_{strategy_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"   Histórico de treinamento salvo: {csv_path}")
    print(f"   Rodadas registradas: {len(df)}")
    
    if 'aucpr_mean' in df.columns:
        print(f"   Intervalo AUCPR: {df['aucpr_mean'].min():.4f} → {df['aucpr_mean'].max():.4f}")
        best_round = int(df['aucpr_mean'].idxmax()) + 1
        print(f"   Melhor AUCPR: {df['aucpr_mean'].max():.4f} (Rodada {best_round})")


@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Lógica principal do servidor com seleção de estratégia.
    
    Args:
        grid: Grade de clientes disponíveis
        context: Contexto da execução com configurações
    """
    
    # Read strategy from config
    strategy_name = context.run_config.get("strategy", "bagging")  # "bagging" or "cyclic"
    
    # Read num_rounds and local_epochs based on strategy
    if strategy_name == "bagging":
        num_rounds = context.run_config["num-server-rounds-bagging"]
    else:
        num_rounds = context.run_config["num-server-rounds-cyclic"]
    
    fraction_train = context.run_config["fraction-train"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]

    # Inicialização do modelo global (vazio no início)
    global_model = b""
    arrays = ArrayRecord([np.frombuffer(global_model, dtype=np.uint8)])

    training_history.clear()
    
    # Seleção de estratégia baseada na configuração
    if strategy_name == "cyclic":
        print("\n" + "="*80)
        print("FEDERATED XGBOOST - TREINAMENTO CÍCLICO")
        print("="*80)
        print(f"   Estratégia: Um cliente por rodada (cliente a cliente)")
        print(f"   Rodadas: {num_rounds}")
        print(f"   Rodadas de boosting local: {context.run_config.get('local-epochs-cyclic', 10)}")
        print("="*80 + "\n")
        
        strategy = FedXgbCyclicWithHistory(
            fraction_evaluate=fraction_evaluate,
        )
    else:
        print("\n" + "="*80)
        print("FEDERATED XGBOOST - AGREGAÇÃO BAGGING")
        print("="*80)
        print(f"   Estratégia: Agregação bootstrap")
        print(f"   Rodadas: {num_rounds}")
        print(f"   Clientes por rodada: {int(fraction_train * 44)}/44 ({fraction_train*100:.0f}%)")
        print(f"   Rodadas de boosting local: {context.run_config.get('local-epochs-bagging', 3)}")
        print("="*80 + "\n")
        
        strategy = FedXgbBaggingWithHistory(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
        )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # Salvamento do modelo final em disco
    bst = xgb.Booster(params=params)
    global_model = bytearray(result.arrays["0"].numpy().tobytes())
    bst.load_model(global_model)

    # Determinação do caminho de salvamento baseado na estratégia
    script_dir = Path(__file__).parent
    models_dir = script_dir.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_json = models_dir / f"global_model_{strategy_name}_final.json"

    print("\n" + "="*80)
    print("SALVANDO MODELO FINAL")
    print("="*80)
    print(f"   Estratégia: {strategy_name}")
    print(f"   Caminho JSON: {model_json}")
    
    bst.save_model(str(model_json))
    
    # Salvamento em formato .pt para compatibilidade com análise SHAP
    pt_model_path = models_dir / f"global_model_{strategy_name}_final.pt"
    print(f"   Formato binário: {pt_model_path}")
    bst.save_model(str(pt_model_path))

    # Salvamento em formato JOBLIB padronizado
    joblib_path = models_dir / f"global_model_{strategy_name}_final.joblib"
    model_data = {
        'best_model': bst,
        'best_params': None,
        'best_score': None,
        'study': None,
        'scaler': None,
        'feature_names': None,
        'train_auc': None,
        'test_auc': None,
        'metadata': {
            'saved_at': datetime.now().isoformat(timespec='seconds'),
            'model_type': 'XGBoost',
            'framework': 'Flower Federated',
            'strategy': strategy_name,
            'num_rounds': num_rounds,
            'params': params,
        }
    }
    joblib.dump(model_data, joblib_path)
    print(f"   JOBLIB salvo: {joblib_path}")
    
    print("   Modelo salvo com sucesso!")
    print("="*80 + "\n")
    
    print("\n" + "="*80)
    print("SALVANDO HISTÓRICO DE TREINAMENTO")
    print("="*80)
    save_training_history(training_history, strategy_name, models_dir)
    print("="*80 + "\n")
