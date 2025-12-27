"""
Análise LIME para XGBoost Federado (estratégia Cyclic).

Gera explicações LIME por instância (CSV único: uma linha por instância)
e visualizações agregadas comparáveis aos outputs SHAP.

Este script espera o modelo federado em:
  flwr-xgboost/models/global_model_cyclic_final.joblib (ou .pt como fallback)
"""
from pathlib import Path
import sys
import argparse
import pickle
import tempfile
import re
import os

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_tabular import LimeTabularExplainer

# Adiciona paths do projeto para importar utils e federated_xgboost
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "flwr-xgboost"))

from utils import training_utils
from utils.shap_utils import load_shap_samples as utils_load_shap_samples, load_model_from_joblib

try:
    from federated_xgboost.task import _encode_wind_directions_cyclic
except Exception:
    # Fallback
    def _encode_wind_directions_cyclic(df):
        return df

def fix_base_score_in_json(json_path: str) -> str:
    """Corrige base_score no JSON (de '[2.2154084E-1]' para '0.22154084'). Retorna path do JSON temporrio fixado."""
    with open(json_path, 'r', encoding='utf8') as f:
        model_text = f.read()

    pattern1 = r'"base_score"\s*:\s*"\[([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\]"'
    pattern2 = r'"base_score"\s*:\s*\[([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\]'

    def _repl(m):
        val = float(m.group(1))
        return f'"base_score": "{val}"'

    model_text_fixed = re.sub(pattern1, _repl, model_text)
    model_text_fixed = re.sub(pattern2, _repl, model_text_fixed)

    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w', encoding='utf8')
    tmpf.write(model_text_fixed)
    tmpf.close()
    return tmpf.name

def load_xgboost_model_fixing_json(model_path: str):
    """Carrega xgboost.Booster de forma robusta, exporta pra JSON, corrige base_score e recarrega."""
    import xgboost as xgb

    tmp_json_1 = None
    tmp_json_2 = None
    try:
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict) and 'model' in data:
                model_bytes = data['model']
                bst_temp = xgb.Booster()
                bst_temp.load_model(bytearray(model_bytes))
            else:
                if isinstance(data, (bytes, bytearray)):
                    bst_temp = xgb.Booster()
                    bst_temp.load_model(bytearray(data))
                else:
                    raise RuntimeError("Pickle content not recognized as booster bytes")
            print("Booster carregado de container pickle")
        except Exception:
            bst_temp = xgb.Booster()
            bst_temp.load_model(model_path)
            print("Booster carregado de path direto")

        # Salva em JSON temp e aplica fix de base_score
        tmp_json_1 = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp_json_1.close()
        bst_temp.save_model(tmp_json_1.name)
        tmp_json_2 = fix_base_score_in_json(tmp_json_1.name)

        bst = xgb.Booster()
        bst.load_model(tmp_json_2)
        print("Booster recarregado de JSON corrigido (base_score fixado)")

        return bst
    finally:
        for p in (tmp_json_1, tmp_json_2):
            try:
                if p is not None:
                    path = p.name if hasattr(p, 'name') else p
                    if path and os.path.exists(path):
                        os.remove(path)
            except Exception:
                pass

def _preprocess_xgb(df_shap: pd.DataFrame):
    df_local = df_shap.copy()
    df_local = df_local.drop(columns=['Location'], errors='ignore')
    df_local = _encode_wind_directions_cyclic(df_local)
    X_df = df_local.drop(columns=['RainTomorrow'])
    y_series = df_local['RainTomorrow']
    feature_names = X_df.columns.tolist()
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_np = imputer.fit_transform(X_df.values)
    y_np = y_series.values
    return X_np, y_np, feature_names

def build_predict_proba_for_booster_or_joblib(artifact, feature_names):
    """Cria predict_proba para xgboost.Booster ou sklearn wrapper. Passa feature_names pro DMatrix."""
    import xgboost as xgb

    if isinstance(artifact, dict) and 'best_model' in artifact:
        candidate = artifact['best_model']
    else:
        candidate = artifact

    # sklearn-like
    if hasattr(candidate, 'predict_proba'):
        def predict_proba(X):
            return candidate.predict_proba(X)
        return predict_proba

    if isinstance(candidate, xgb.Booster) or 'booster' in str(type(candidate)).lower():
        booster = candidate if isinstance(candidate, xgb.Booster) else load_xgboost_model_fixing_json(candidate)
        def predict_proba(X):
            Xnp = np.asarray(X)
            dmat = xgb.DMatrix(Xnp, feature_names=feature_names)
            preds = booster.predict(dmat)
            preds = np.asarray(preds)
            if preds.ndim == 1:
                probs1 = preds
                return np.vstack([1 - probs1, probs1]).T
            elif preds.ndim == 2:
                return preds
            else:
                return preds.reshape((preds.shape[0], -1))
        return predict_proba

    if isinstance(artifact, (bytes, bytearray, str, Path)):
        try:
            bst = load_xgboost_model_fixing_json(str(artifact))
            return build_predict_proba_for_booster_or_joblib(bst, feature_names)
        except Exception:
            pass

    if hasattr(candidate, 'predict'):
        def predict_proba(X):
            preds = candidate.predict(X)
            preds = np.asarray(preds)
            if preds.ndim == 1:
                probs1 = preds
                return np.vstack([1 - probs1, probs1]).T
            elif preds.ndim == 2:
                return preds
            else:
                return preds.reshape((preds.shape[0], -1))
        return predict_proba

    raise RuntimeError("Não foi possível construir predict_proba para o artefato XGBoost (Cyclic).")

def explain_instances_xgb_cyclic(args):
    data_path = project_root / 'datasets' / 'rain_australia' / 'weatherAUS_cleaned.csv'
    shap_indices_path = project_root / 'datasets' / 'shap_sample_indices.csv'
    train_idx_path = project_root / 'datasets' / 'train_indices.csv'

    joblib_path = project_root / 'flwr-xgboost' / 'models' / 'global_model_cyclic_final.joblib'
    legacy_path = project_root / 'flwr-xgboost' / 'models' / 'global_model_cyclic_final.pt'

    model_artifact = None
    if joblib_path.exists():
        model_artifact = load_model_from_joblib(joblib_path)
        if isinstance(model_artifact, dict) and 'best_model' in model_artifact:
            candidate = model_artifact['best_model']
        else:
            candidate = model_artifact
    elif legacy_path.exists():
        try:
            with open(str(legacy_path), 'rb') as f:
                model_artifact = pickle.load(f)
                candidate = model_artifact
        except Exception:
            candidate = load_xgboost_model_fixing_json(str(legacy_path))
            model_artifact = candidate
    else:
        raise FileNotFoundError(f"Modelo XGBoost (Cyclic) não encontrado em {joblib_path} nem {legacy_path}")

    X_pool, y_pool, feature_names = utils_load_shap_samples(
        Path(data_path), Path(shap_indices_path), preprocess_fn=_preprocess_xgb
    )

    df, train_indices, _ = training_utils.load_train_test_data(data_path, train_idx_path, project_root / 'datasets' / 'test_indices.csv')
    X_train_trans, _, feature_names_train = _preprocess_xgb(df.loc[train_indices])
    feature_names_for_model = feature_names_train

    # Cria predict_proba passando feature_names pro DMatrix
    predict_proba = build_predict_proba_for_booster_or_joblib(candidate, feature_names_for_model)

    explainer = LimeTabularExplainer(
        training_data=X_train_trans,
        feature_names=feature_names_for_model,
        class_names=['NoRain', 'Rain'],
        discretize_continuous=False,
        random_state=args.random_state
    )

    # Amostra balanceada entre classes
    rng = np.random.RandomState(args.random_state)
    n_pool = len(X_pool)
    if args.sample_count >= n_pool:
        chosen_local = np.arange(n_pool)
    else:
        g0 = np.where(y_pool == 0)[0]
        g1 = np.where(y_pool == 1)[0]
        n0 = args.sample_count // 2
        n1 = args.sample_count - n0
        c0 = rng.choice(g0, size=min(n0, len(g0)), replace=False) if len(g0) > 0 else np.array([], dtype=int)
        c1 = rng.choice(g1, size=min(n1, len(g1)), replace=False) if len(g1) > 0 else np.array([], dtype=int)
        chosen_local = np.concatenate([c0, c1])

    shap_df = pd.read_csv(shap_indices_path)
    pool_indices = shap_df['index'].values
    chosen = pool_indices[chosen_local]

    out_dir = project_root / 'xAI' / 'lime_results_federated' / 'xgboost' / 'cyclic_strategy'
    out_dir.mkdir(parents=True, exist_ok=True)

    per_instance_records = []

    print(f"Gerando explicações LIME para {len(chosen_local)} instâncias (XGBoost federado - cyclic)...")
    for local_idx, orig_idx in zip(chosen_local, chosen):
        x_row_trans = X_pool[local_idx].reshape(1, -1)
        explanation = explainer.explain_instance(
            x_row_trans[0],
            predict_proba,
            num_features=args.num_features,
            num_samples=args.num_samples,
            labels=(1,)
        )
        elist = explanation.as_list(label=1)
        recs = []
        for feat, weight in elist:
            recs.append({
                'index': int(orig_idx),
                'feature': feat,
                'weight': float(weight),
                'abs_weight': abs(float(weight)),
                'sign': 'pos' if weight > 0 else 'neg',
                'true_label': int(y_pool[local_idx])
            })
        per_inst_df = pd.DataFrame(recs)
        per_instance_records.append(per_inst_df)

    if len(per_instance_records) == 0:
        print("Nenhuma explicação gerada. Saindo.")
        return

    all_df = pd.concat(per_instance_records, ignore_index=True)

    # Cria CSV com uma linha por instância (features com pesos signed)
    instance_wide = all_df.pivot_table(index='index', columns='feature', values='weight', aggfunc='sum').fillna(0)
    label_map = all_df.groupby('index')['true_label'].first()
    instance_wide = instance_wide.merge(label_map.rename('true_label'), left_index=True, right_index=True)
    instance_wide['sum_abs'] = all_df.groupby('index')['abs_weight'].sum().reindex(instance_wide.index).fillna(0)
    instance_wide.reset_index().to_csv(out_dir / 'lime_instance_weights.csv', index=False)

    instance_wide_abs = all_df.pivot_table(index='index', columns='feature', values='abs_weight', aggfunc='sum').fillna(0)
    instance_wide_abs = instance_wide_abs.merge(label_map.rename('true_label'), left_index=True, right_index=True)
    instance_wide_abs['sum_abs'] = all_df.groupby('index')['abs_weight'].sum().reindex(instance_wide_abs.index).fillna(0)
    instance_wide_abs.reset_index().to_csv(out_dir / 'lime_instance_abs_weights.csv', index=False)

    # Importância global: média dos pesos absolutos por feature
    feat_all = all_df.groupby('feature')['abs_weight'].mean().reset_index().rename(columns={'abs_weight': 'mean_abs_weight'})
    feat_all = feat_all.sort_values('mean_abs_weight', ascending=False)
    feat_all.to_csv(out_dir / 'feature_importance_all.csv', index=False)

    for cls in [0, 1]:
        cls_df = all_df[all_df['true_label'] == cls]
        if len(cls_df) == 0:
            continue
        feat_cls = cls_df.groupby('feature')['abs_weight'].mean().reset_index().rename(columns={'abs_weight': 'mean_abs_weight'})
        feat_cls = feat_cls.sort_values('mean_abs_weight', ascending=False)
        feat_cls.to_csv(out_dir / f'feature_importance_class_{cls}.csv', index=False)

    try:
        f0 = pd.read_csv(out_dir / 'feature_importance_class_0.csv')
        f1 = pd.read_csv(out_dir / 'feature_importance_class_1.csv')
        comp = f0.merge(f1, on='feature', how='outer', suffixes=('_0', '_1')).fillna(0)
        comp['diff_0_1'] = comp['mean_abs_weight_0'] - comp['mean_abs_weight_1']
        comp.to_csv(out_dir / 'comparison_class_0_vs_1.csv', index=False)
    except Exception:
        comp = None

    import matplotlib
    matplotlib.use('Agg')

    TOP_K = min(20, len(feat_all))
    top_features = feat_all['feature'].iloc[:TOP_K].tolist()

    plt.figure(figsize=(10, max(4, TOP_K * 0.35)))
    sns.barplot(x='mean_abs_weight', y='feature', data=feat_all.head(TOP_K))
    plt.title('LIME - Importância Global (mean |coef|) - XGBoost Federado (Cyclic) - Top {}'.format(TOP_K))
    plt.tight_layout()
    plt.savefig(out_dir / f'lime_xgb_cyclic_global_mean_abs_weight_top{TOP_K}.png')
    plt.close()

    for cls in [0, 1]:
        csv_path = out_dir / f'feature_importance_class_{cls}.csv'
        if not Path(csv_path).exists():
            continue
        dfc = pd.read_csv(csv_path)
        TOP_K_C = min(20, len(dfc))
        plt.figure(figsize=(10, max(4, TOP_K_C * 0.35)))
        sns.barplot(x='mean_abs_weight', y='feature', data=dfc.head(TOP_K_C))
        plt.title(f'LIME XGBoost Cyclic - Importância Classe {cls} (mean |coef|) - Top {TOP_K_C}')
        plt.tight_layout()
        plt.savefig(out_dir / f'lime_xgb_cyclic_class_{cls}_mean_abs_weight_top{TOP_K_C}.png')
        plt.close()

    df_top = all_df[all_df['feature'].isin(top_features)].copy()
    plt.figure(figsize=(12, max(4, len(top_features) * 0.25)))
    sns.boxplot(x='weight', y='feature', hue='true_label', data=df_top, showfliers=False)
    plt.legend(title='true_label')
    plt.title('Distribuição dos coeficientes LIME (signed) por feature (Top {}) - XGBoost Cyclic'.format(TOP_K))
    plt.tight_layout()
    plt.savefig(out_dir / f'lime_xgb_cyclic_feature_weight_distribution_top{TOP_K}.png')
    plt.close()

    pivot = all_df.pivot_table(index='index', columns='feature', values='weight', aggfunc='sum').fillna(0)
    common_cols = [c for c in top_features if c in pivot.columns]
    heatmat = pivot[common_cols].copy()
    row_labels = []
    for idx in heatmat.index:
        rec = all_df[all_df['index'] == idx]
        row_labels.append(rec['true_label'].iloc[0] if len(rec) > 0 else np.nan)
    heatmat['__label__'] = row_labels
    heatmat['__sum_abs__'] = heatmat[common_cols].abs().sum(axis=1)
    heatmat = heatmat.sort_values(['__label__', '__sum_abs__'], ascending=[True, False])
    lbls = heatmat['__label__'].astype(int).astype(str).values
    heatmat = heatmat[common_cols]

    MAX_ROWS = 300
    if heatmat.shape[0] > MAX_ROWS:
        sampled = heatmat.head(MAX_ROWS)
        sampled_lbls = lbls[:MAX_ROWS]
    else:
        sampled = heatmat
        sampled_lbls = lbls
    plt.figure(figsize=(max(6, len(common_cols) * 0.6), max(6, sampled.shape[0] * 0.06)))
    sns.heatmap(sampled.values, cmap='RdBu_r', center=0, yticklabels=sampled_lbls, xticklabels=common_cols)
    plt.xlabel('Feature')
    plt.ylabel('Instância (rotulado)')
    plt.title(f'Heatmap das contribuições LIME (signed) - XGBoost Cyclic - Top {len(common_cols)} features')
    plt.tight_layout()
    plt.savefig(out_dir / f'lime_xgb_cyclic_heatmap_top{len(common_cols)}_rows{sampled.shape[0]}.png')
    plt.close()

    # Seleciona uma instância representativa de cada classe (maior soma de contribuições)
    rep_instances = []
    sumabs_by_idx = all_df.groupby('index')['abs_weight'].sum().reset_index().rename(columns={'abs_weight': 'sum_abs'})
    label_map_simple = all_df.groupby('index')['true_label'].first().to_dict()
    for cls in [0, 1]:
        cls_idxs = sumabs_by_idx[sumabs_by_idx['index'].isin([i for i, v in label_map_simple.items() if v == cls])]
        if len(cls_idxs) == 0:
            continue
        pick_idx = cls_idxs.sort_values('sum_abs', ascending=False)['index'].iloc[0]
        rep_instances.append((cls, int(pick_idx)))
    for cls, inst_idx in rep_instances:
        inst_df = all_df[all_df['index'] == inst_idx].copy().sort_values('weight', ascending=True)
        plt.figure(figsize=(8, max(3, len(inst_df) * 0.4)))
        colors = inst_df['sign'].map({'pos': '#2ca02c', 'neg': '#d62728'})
        plt.barh(inst_df['feature'], inst_df['weight'], color=colors)
        plt.xlabel('contribuição (weight)')
        plt.title(f'LIME - Waterfall-like contributions (idx={inst_idx}, true_label={cls}) - XGBoost Cyclic')
        plt.tight_layout()
        plt.savefig(out_dir / f'lime_xgb_cyclic_waterfall_idx{inst_idx}_label{cls}.png')
        plt.close()

    summary_lines = []
    summary_lines.append('# Resumo LIME - XGBoost Federado (Cyclic)\n')
    summary_lines.append(f'Instâncias explicadas (sample_count): {len(chosen_local)}\n')
    summary_lines.append(f'Top {TOP_K} features (global, mean |weight|):\n')
    for i, row in feat_all.head(TOP_K).iterrows():
        summary_lines.append(f"- {row['feature']}: {row['mean_abs_weight']:.6f}\n")
    if comp is not None:
        summary_lines.append('\nDiferenças mean |weight| (classe 0 - classe 1) (top 10 por abs diff):\n')
        comp['abs_diff'] = comp['diff_0_1'].abs()
        for _, r in comp.sort_values('abs_diff', ascending=False).head(10).iterrows():
            summary_lines.append(f"- {r['feature']}: classe0={r['mean_abs_weight_0']:.6f}, classe1={r['mean_abs_weight_1']:.6f}, diff={r['diff_0_1']:.6f}\n")
    summary_lines.append('\nArquivos gerados:\n')
    for f in sorted([p.name for p in out_dir.iterdir()]):
        summary_lines.append(f"- {f}\n")
    with open(out_dir / 'summary.md', 'w', encoding='utf8') as fh:
        fh.writelines(summary_lines)

    print(f"Resultados LIME salvos em: {out_dir}")
    

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sample-count', type=int, default=2000, help='Number of instances to explain')
    p.add_argument('--num-features', type=int, default=19, help='Top K features to show per instance')
    p.add_argument('--num-samples', type=int, default=500, help='Number of perturbed samples per explanation')
    p.add_argument('--random-state', type=int, default=42)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    explain_instances_xgb_cyclic(args)
