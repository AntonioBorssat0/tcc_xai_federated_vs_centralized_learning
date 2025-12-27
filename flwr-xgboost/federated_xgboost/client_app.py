"""
Cliente Flower para treinamento federado com XGBoost.
Suporta estratégias Bagging e Cyclic.
"""

import warnings
import numpy as np
import xgboost as xgb
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict

from federated_xgboost.task import load_data, replace_keys

warnings.filterwarnings("ignore", category=UserWarning)


app = ClientApp()


def _local_boost(bst_input, num_local_round, train_dmatrix):
    """
    Executa rodadas locais de boosting.
    
    Args:
        bst_input: Modelo XGBoost base
        num_local_round: Número de árvores a treinar
        train_dmatrix: Dados de treinamento
    
    Returns:
        Modelo com as últimas N árvores (para agregação no servidor)
    """
    for i in range(num_local_round):
        bst_input.update(train_dmatrix, bst_input.num_boosted_rounds())

    # Extrai as últimas N árvores para agregação no servidor
    bst = bst_input[
        bst_input.num_boosted_rounds()
        - num_local_round : bst_input.num_boosted_rounds()
    ]
    return bst


@app.train()
def train(msg: Message, context: Context) -> Message:
    """
    Treina modelo XGBoost local.
    
    Args:
        msg: Mensagem do servidor com modelo global
        context: Contexto da execução com configurações
    
    Returns:
        Mensagem com modelo treinado e métricas
    """
    partition_id = context.node_config["partition-id"]
    
    keep_location = context.run_config.get("keep-location", False)
    strategy = context.run_config.get("strategy", "bagging")
    
    # Carregamento dos dados da partição (localização geográfica)
    train_dmatrix, _, num_train, _ = load_data(
        partition_id=partition_id,
        keep_location=keep_location
    )

    strategy = context.run_config.get("strategy", "bagging")
    
    # Número de épocas locais depende da estratégia
    if strategy == "bagging":
        num_local_round = context.run_config["local-epochs-bagging"]
    else:
        num_local_round = context.run_config["local-epochs-cyclic"]
    
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]

    global_round = msg.content["config"]["server-round"]
    
    if global_round == 1:
        # Primeira rodada: treina do zero
        bst = xgb.train(
            params,
            train_dmatrix,
            num_boost_round=num_local_round,
        )
    else:
        # Carrega modelo global recebido do servidor
        bst = xgb.Booster(params=params)
        global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
        bst.load_model(global_model)

        if strategy == "bagging":
            # Bagging: treina novas árvores e extrai apenas as últimas N
            bst = _local_boost(bst, num_local_round, train_dmatrix)
        else:
            # Cyclic: continua boosting a partir do modelo global
            for i in range(num_local_round):
                bst.update(train_dmatrix, bst.num_boosted_rounds())

    # Serialização do modelo para envio ao servidor
    local_model = bst.save_raw("json")
    model_np = np.frombuffer(local_model, dtype=np.uint8)

    # Construção da mensagem de resposta
    model_record = ArrayRecord([model_np])
    metrics = {
        "num-examples": num_train,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """
    Avalia modelo XGBoost global.
    
    Args:
        msg: Mensagem do servidor com modelo global
        context: Contexto da execução com configurações
    
    Returns:
        Mensagem com métricas de avaliação
    """
    partition_id = context.node_config["partition-id"]
    
    keep_location = context.run_config.get("keep-location", False)
    
    # Carregamento dos dados de validação da partição
    _, valid_dmatrix, _, num_val = load_data(
        partition_id=partition_id,
        keep_location=keep_location
    )

    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]

    # Carregamento do modelo global
    bst = xgb.Booster(params=params)
    global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
    bst.load_model(global_model)

    # Avaliação do modelo
    eval_results = bst.eval_set(
        evals=[(valid_dmatrix, "valid")],
        iteration=bst.num_boosted_rounds() - 1,
    )
    # Parsing do resultado: "valid\taucpr:0.1234"
    aucpr = float(eval_results.split("\t")[1].split(":")[1])

    # Construção da mensagem de resposta
    metrics = {
        "aucpr": aucpr,
        "num-examples": num_val,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
