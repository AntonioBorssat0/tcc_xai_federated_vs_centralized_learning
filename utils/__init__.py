"""
Funções utilitárias para o projeto.
Código compartilhado para treinamento e análise SHAP.
"""

from .training_utils import (
    load_train_test_data,
    save_model_joblib,
    compute_metrics,
    print_section,
    print_metrics,
    print_class_distribution
)

from .shap_utils import (
    load_model_from_joblib,
    load_shap_samples,
    create_shap_visualizations,
    save_feature_importance_csv,
    print_top_features
)


__all__ = [
    # Utilitários de treinamento
    'load_train_test_data',
    'save_model_joblib',
    'compute_metrics',
    'print_section',
    'print_metrics',
    'print_class_distribution',
    
    # Utilitários SHAP
    'load_model_from_joblib',
    'load_shap_samples',
    'create_shap_visualizations',
    'save_feature_importance_csv',
    'print_top_features',
    
]
