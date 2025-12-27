"""
Script utilitário para carregar o melhor checkpoint do treinamento federado MLP.

Analisa o histórico de treinamento para identificar a melhor rodada
baseado no AUCPR de validação e carrega o checkpoint correspondente.
"""

import json
import torch
from pathlib import Path
from task import WeatherMLP


def load_best_checkpoint(models_dir: Path = Path("models")):
    """
    Carrega o melhor checkpoint baseado no histórico de treinamento.
    
    Args:
        models_dir: Diretório contendo models/ e checkpoints/
        
    Returns:
        tuple: (melhor_modelo, melhor_rodada, melhores_métricas)
    """
    # Carregamento do histórico de treinamento
    history_path = models_dir / "training_history.json"
    if not history_path.exists():
        raise FileNotFoundError(
            f"Histórico de treinamento não encontrado: {history_path}\n"
            "Por favor, execute o treinamento federado primeiro."
        )
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Busca pela melhor rodada baseada em val_aucpr
    best_idx = None
    best_aucpr = -1.0
    
    for i, record in enumerate(history):
        val_aucpr = record.get('val_aucpr')
        if val_aucpr is not None and val_aucpr > best_aucpr:
            best_aucpr = val_aucpr
            best_idx = i
    
    if best_idx is None:
        raise ValueError("Nenhum val_aucpr válido encontrado no histórico de treinamento")
    
    best_record = history[best_idx]
    best_round = best_record['round']
    
    print("="*80)
    print("CARREGANDO MELHOR CHECKPOINT")
    print("="*80)
    print(f"\nMelhor Rodada: {best_round}")
    print(f"Val AUCPR:  {best_record.get('val_aucpr', 'N/A'):.6f}")
    print(f"Val AUC:    {best_record.get('val_auc', 'N/A'):.6f}")
    print(f"Val Acc:    {best_record.get('val_accuracy', 'N/A'):.6f}")
    
    # Carregamento do checkpoint
    checkpoint_path = models_dir / "checkpoints" / f"round_{best_round:03d}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {checkpoint_path}\n"
            "Os checkpoints podem ter sido deletados ou o treinamento não foi concluído."
        )
    
    print(f"\nCarregando: {checkpoint_path.absolute()}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Recria o modelo com os hiperparâmetros salvos
    model = WeatherMLP(
        input_size=checkpoint['input_size'],
        hidden1=checkpoint['hidden1'],
        hidden2=checkpoint['hidden2'],
        hidden3=checkpoint['hidden3'],
        dropout1=checkpoint['dropout1'],
        dropout2=checkpoint['dropout2']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Modelo carregado com sucesso")
    
    # Exibição da arquitetura
    print(f"\nArquitetura:")
    print(f"  Input:  {checkpoint['input_size']}")
    print(f"  Hidden: [{checkpoint['hidden1']}, {checkpoint['hidden2']}, {checkpoint['hidden3']}]")
    print(f"  Dropout: [{checkpoint['dropout1']:.3f}, {checkpoint['dropout2']:.3f}]")
    
    # Cálculo de rodadas economizadas
    total_rounds = checkpoint['num_rounds']
    rounds_saved = total_rounds - best_round
    if rounds_saved > 0:
        pct_saved = (rounds_saved / total_rounds) * 100
        print(f"\nEficiência:")
        print(f"  Rodadas economizadas: {rounds_saved}/{total_rounds} ({pct_saved:.1f}%)")
    
    print("="*80 + "\n")
    
    return model, best_round, best_record


def save_best_model(model, best_round, best_metrics, output_path: Path = Path("models/best_model.pt")):
    """
    Salva o melhor modelo como um artefato independente.
    
    Args:
        model: O modelo a ser salvo
        best_round: Número da rodada onde o melhor desempenho foi alcançado
        best_metrics: Dicionário de métricas para essa rodada
        output_path: Onde salvar o modelo
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_round': best_round,
        'best_metrics': best_metrics,
        'model_config': {
            'input_size': model.input_size,
            'hidden1': model.hidden1,
            'hidden2': model.hidden2,
            'hidden3': model.hidden3,
            'dropout1': model.dropout1,
            'dropout2': model.dropout2,
        }
    }, output_path)
    
    print(f"Melhor modelo salvo: {output_path.absolute()}")


if __name__ == "__main__":
    # Carrega o melhor checkpoint
    model, best_round, best_metrics = load_best_checkpoint()
    
    # Salva como artefato independente
    save_best_model(model, best_round, best_metrics)
    
    print("\nConcluído! Use este modelo para inferência/avaliação.")
