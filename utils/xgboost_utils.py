"""Utilitários específicos para XGBoost no Projeto.
Trata bug do base_score e carregamento de modelos XGBoost.
"""

import xgboost as xgb
import tempfile
import os
import json
import re
import pickle
import numpy as np
from pathlib import Path
from typing import Optional


def fix_base_score_in_json(json_path: str) -> str:
    """
    Corrige o campo base_score no arquivo JSON do modelo XGBoost.
    Converte formato '[2.2154084E-1]' para formato '0.22154084'.
    
    Args:
        json_path: Caminho para arquivo JSON com base_score potencialmente quebrado
    
    Returns:
        Caminho para arquivo JSON corrigido (temporário)
    """
    with open(json_path, 'r') as f:
        model_text = f.read()
    
    # Padrão 1: "base_score": "[number]"
    pattern1 = r'"base_score"\s*:\s*"\[([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\]"'
    
    def replace_base_score(match):
        original = match.group(0)
        value_str = match.group(1)
        value_float = float(value_str)
        replacement = f'"base_score": "{value_float}"'
        print(f"   Corrigido base_score: {original} → {replacement}")
        return replacement
    
    model_text_fixed = re.sub(pattern1, replace_base_score, model_text)
    
    # Padrão 2: "base_score": [number] (sem aspas ao redor do array)
    pattern2 = r'"base_score"\s*:\s*\[([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\]'
    
    def replace_base_score_array(match):
        original = match.group(0)
        value_str = match.group(1)
        value_float = float(value_str)
        replacement = f'"base_score": "{value_float}"'
        print(f"   Corrigido base_score (array): {original} → {replacement}")
        return replacement
    
    model_text_fixed = re.sub(pattern2, replace_base_score_array, model_text_fixed)
    
    # Salvar JSON corrigido em arquivo temporário
    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w')
    tmpf.write(model_text_fixed)
    tmpf.close()
    
    return tmpf.name


def load_xgboost_model_safe(
    model_path: Path,
    model_type: str = "centralized"
) -> xgb.Booster:
    """
    Carrega modelo XGBoost com correção robusta de base_score.
    Funciona para modelos centralizados (.json/.pkl) e federados (.pt).
    
    Args:
        model_path: Caminho para arquivo do modelo
        model_type: "centralized", "federated_cyclic", ou "federated_bagging"
    
    Returns:
        Instância xgb.Booster pronta para predição e SHAP
    """
    print(f"Carregando modelo XGBoost ({model_type})...")
    print(f"   Arquivo: {model_path}")
    
    tmp_json_1 = None
    tmp_json_2 = None
    
    try:
        # Etapa 1: Carregar modelo baseado no formato
        bst_temp = None
        
        if str(model_path).endswith('.pt'):
            # Modelo federado (pickle com dict)
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                if isinstance(model_data, dict) and 'model' in model_data:
                    model_bytes = model_data['model']
                else:
                    model_bytes = model_data
                
                bst_temp = xgb.Booster()
                bst_temp.load_model(bytearray(model_bytes))
                print(f"   Modelo carregado de .pt (pickle)")
            except Exception as e:
                raise RuntimeError(f"Erro ao carregar .pt: {e}")
        
        elif str(model_path).endswith('.pkl'):
            # Tentar pickle primeiro
            try:
                with open(model_path, 'rb') as f:
                    bst_temp = pickle.load(f)
                print(f"   Modelo carregado de .pkl")
            except Exception as e:
                raise RuntimeError(f"Erro ao carregar .pkl: {e}")
        
        elif str(model_path).endswith('.json'):
            # Carregamento direto de JSON
            bst_temp = xgb.Booster()
            bst_temp.load_model(str(model_path))
            print(f"   Modelo carregado de .json")
        
        else:
            raise ValueError(f"Formato de arquivo não suportado: {model_path}")
        
        # Etapa 2: SEMPRE aplicar correção de base_score
        print(f"   Aplicando correção de base_score...")
        tmp_json_1 = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp_json_1.close()
        bst_temp.save_model(tmp_json_1.name)
        
        # Corrigir base_score
        tmp_json_2 = fix_base_score_in_json(tmp_json_1.name)
        
        # Etapa 3: Carregar modelo corrigido
        bst = xgb.Booster()
        bst.load_model(tmp_json_2)
        print(f"   Modelo carregado com base_score corrigido")
        
        # Mostrar informações
        try:
            print(f"   - Número de features: {bst.num_features()}")
        except Exception:
            pass
        
        return bst
        
    finally:
        # Limpar arquivos temporários
        for tmp_file in [tmp_json_1, tmp_json_2]:
            if tmp_file is not None:
                tmp_path = tmp_file.name if hasattr(tmp_file, 'name') else tmp_file
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass


def save_xgboost_model_safe(
    bst: xgb.Booster,
    save_path: Path,
    fix_base_score: bool = True
) -> None:
    """
    Salva modelo XGBoost com base_score já corrigido.
    
    Args:
        bst: XGBoost Booster
        save_path: Caminho para salvar modelo
        fix_base_score: Se deve corrigir base_score antes de salvar
    """
    if fix_base_score:
        # Salvar em temp, corrigir, depois salvar na localização final
        tmp_json = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp_json.close()
        
        try:
            # Salvar em temp
            bst.save_model(tmp_json.name)
            
            # Corrigir base_score
            tmp_fixed = fix_base_score_in_json(tmp_json.name)
            
            # Carregar modelo corrigido
            bst_fixed = xgb.Booster()
            bst_fixed.load_model(tmp_fixed)
            
            # Salvar na localização final
            bst_fixed.save_model(str(save_path))
            
            # Limpar
            os.remove(tmp_json.name)
            os.remove(tmp_fixed)
            
        except Exception as e:
            # Fallback: salvar sem correção
            print(f"   [AVISO] Erro ao corrigir base_score: {e}")
            print(f"   [AVISO] Salvando sem correção...")
            bst.save_model(str(save_path))
    else:
        bst.save_model(str(save_path))
