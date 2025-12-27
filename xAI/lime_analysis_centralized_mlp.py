"""
Análise LIME para MLP Centralizado.

Produz explicações LIME por instância e importâncias de features agregadas.

Este script espera o modelo centralizado salvo como joblib em:
  centralized_training/models/mlp/centralized_model_best.joblib
"""
from pathlib import Path
import sys
import argparse
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from lime.lime_tabular import LimeTabularExplainer

# Adiciona paths do projeto para importar utils e federated_mlp
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
flwr_mlp_path = project_root / 'flwr-mlp'
if str(flwr_mlp_path) not in sys.path and flwr_mlp_path.exists():
    sys.path.insert(0, str(flwr_mlp_path))

from utils import training_utils
from utils.shap_utils import load_model_from_joblib, load_shap_samples as utils_load_shap_samples
try:
    from federated_mlp.task import prepare_weather_data, LABEL_COL
except Exception:
    # Fallback
    prepare_weather_data = None
    LABEL_COL = 'RainTomorrow'


def load_model(model_path: Path):
    try:
        data = joblib.load(model_path)
    except ModuleNotFoundError as e:
        # Tenta adicionar paths dos pacotes federados antes de desistir
        candidates = [project_root / 'flwr-mlp', project_root / 'flwr-xgboost']
        for c in candidates:
            cstr = str(c)
            if cstr not in sys.path and c.exists():
                sys.path.insert(0, cstr)
        try:
            data = joblib.load(model_path)
        except Exception:
            raise ModuleNotFoundError(
                f"Falha ao carregar {model_path}. Módulo ausente: {e}.\n"
                + "Tentou adicionar flwr-mlp e flwr-xgboost ao sys.path. "
                + "Se o joblib refere-se a um módulo customizado, certifique-se que a pasta do pacote está disponível."
            ) from e
    # Modelo pode estar dentro de dict ou salvo diretamente
    if isinstance(data, dict) and 'best_model' in data:
        model = data['best_model']
        scaler = data.get('scaler')
        feature_names = data.get('feature_names')
    else:
        model = data
        scaler = None
        feature_names = None
    return model, scaler, feature_names


def build_predict_fn(model, scaler, feature_names):
    import inspect
    try:
        if hasattr(model, 'predict_proba'):
            def predict_proba(X):
                if scaler is not None:
                    Xt = scaler.transform(X)
                else:
                    Xt = X
                return model.predict_proba(Xt)
            return predict_proba
    except Exception:
        pass

    try:
        import torch
        if hasattr(model, 'forward') or 'torch' in str(type(model)):
            device = torch.device('cpu')

            def predict_proba(X):
                if scaler is not None:
                    Xt = scaler.transform(X)
                else:
                    Xt = X
                tensor = torch.tensor(Xt, dtype=torch.float32).to(device)
                model.eval()
                with torch.no_grad():
                    out = model(tensor)
                    out_cpu = out.cpu()
                    # Lida com diferentes formatos de output do modelo
                    if out_cpu.dim() == 2:
                        ncols = out_cpu.size(1)
                        if ncols == 1:
                            probs1 = torch.sigmoid(out_cpu.squeeze(1)).numpy()
                            probs = np.vstack([1 - probs1, probs1]).T
                        else:
                            probs = torch.softmax(out_cpu, dim=1).numpy()
                    elif out_cpu.dim() == 1:
                        probs1 = torch.sigmoid(out_cpu).numpy()
                        probs = np.vstack([1 - probs1, probs1]).T
                    else:
                        try:
                            reshaped = out_cpu.view(out_cpu.size(0), -1)
                            probs = torch.softmax(reshaped, dim=1).numpy()
                        except Exception:
                            probs = np.zeros((Xt.shape[0], 2), dtype=float)
                return probs

            return predict_proba
    except Exception:
        pass

    if hasattr(model, 'predict'):
        def predict_proba(X):
            if scaler is not None:
                Xt = scaler.transform(X)
            else:
                Xt = X
            preds = model.predict(Xt)
            probs = np.vstack([1 - preds, preds]).T
            return probs
        return predict_proba

    raise RuntimeError("Modelo não expõe API de predição utilizável (predict_proba/predict ou não é modelo torch)")


def explain_instances(args):
    project_root = Path(__file__).resolve().parent.parent

    data_path = project_root / 'datasets' / 'rain_australia' / 'weatherAUS_cleaned.csv'
    train_idx_path = project_root / 'datasets' / 'train_indices.csv'
    test_idx_path = project_root / 'datasets' / 'test_indices.csv'
    shap_indices_path = project_root / 'datasets' / 'shap_sample_indices.csv'

    df, train_indices, test_indices = training_utils.load_train_test_data(
        data_path, train_idx_path, test_idx_path
    )

    model_path = project_root / 'centralized_training' / 'models' / 'mlp' / 'centralized_model_best.joblib'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model_data = load_model_from_joblib(model_path)
    model = model_data.get('best_model')
    scaler = model_data.get('scaler')

    def preprocess_mlp(df_shap: pd.DataFrame):
        df_processed = prepare_weather_data(df_shap.copy(), use_location=False) if prepare_weather_data is not None else df_shap.copy()
        X_df = df_processed.drop(columns=[LABEL_COL])
        y_series = df_processed[LABEL_COL]
        feature_names_local = X_df.columns.tolist()
        X_np = X_df.values
        y_np = y_series.values
        if scaler is not None:
            X_scaled = scaler.transform(X_np)
        else:
            X_scaled = X_np
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return X_scaled, y_np, feature_names_local

    X_pool, y_pool, feature_names = utils_load_shap_samples(
        project_root / 'datasets' / 'rain_australia' / 'weatherAUS_cleaned.csv',
        project_root / 'datasets' / 'shap_sample_indices.csv',
        preprocess_fn=preprocess_mlp
    )

    shap_df = pd.read_csv(shap_indices_path)
    pool_indices = shap_df['index'].values

    rng = np.random.RandomState(args.random_state)
    n_pool = len(X_pool)
    if args.sample_count >= n_pool:
        chosen_local = np.arange(n_pool)
    else:
        # Amostra balanceada entre classes
        group0 = np.where(y_pool == 0)[0]
        group1 = np.where(y_pool == 1)[0]
        n0 = args.sample_count // 2
        n1 = args.sample_count - n0
        chosen0 = rng.choice(group0, size=min(n0, len(group0)), replace=False) if len(group0) > 0 else np.array([], dtype=int)
        chosen1 = rng.choice(group1, size=min(n1, len(group1)), replace=False) if len(group1) > 0 else np.array([], dtype=int)
        chosen_local = np.concatenate([chosen0, chosen1])

    # Mapeia índices locais do pool para índices originais do dataset
    chosen = pool_indices[chosen_local]

    # Background do explainer usa dados de treino (mesmo preprocessamento do SHAP)
    X_train_trans, _, feature_names_train = preprocess_mlp(df.loc[train_indices])
    feature_names = feature_names_train

    # Scaler já aplicado no preprocessing, então predict_fn recebe None
    predict_proba = build_predict_fn(model, None, feature_names)

    explainer = LimeTabularExplainer(
        training_data=X_train_trans,
        feature_names=feature_names,
        class_names=['NoRain', 'Rain'],
        discretize_continuous=False,
        random_state=args.random_state
    )

    out_dir = project_root / 'xAI' / 'lime_results_centralized' / 'mlp'
    out_dir.mkdir(parents=True, exist_ok=True)

    per_instance_records = []

    print(f"Gerando explicações LIME para {len(chosen_local)} instâncias...")
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
            recs.append({'index': int(orig_idx), 'feature': feat, 'weight': float(weight), 'abs_weight': abs(float(weight)), 'sign': 'pos' if weight > 0 else 'neg', 'true_label': int(y_pool[local_idx])})

        per_inst_df = pd.DataFrame(recs)
        per_instance_records.append(per_inst_df)

    if len(per_instance_records) == 0:
        print("Nenhuma explicação foi gerada. Saindo.")
        return

    all_df = pd.concat(per_instance_records, ignore_index=True)

    # Cria CSV com uma linha por instância (colunas = features com pesos signed)
    instance_wide = all_df.pivot_table(
        index='index', columns='feature', values='weight', aggfunc='sum'
    ).fillna(0)

    label_map = all_df.groupby('index')['true_label'].first()
    instance_wide = instance_wide.merge(label_map.rename('true_label'), left_index=True, right_index=True)
    instance_wide['sum_abs'] = all_df.groupby('index')['abs_weight'].sum().reindex(instance_wide.index).fillna(0)

    instance_csv_path = out_dir / 'lime_instance_weights.csv'
    instance_wide.reset_index().to_csv(instance_csv_path, index=False)

    instance_wide_abs = all_df.pivot_table(
        index='index', columns='feature', values='abs_weight', aggfunc='sum'
    ).fillna(0)
    instance_wide_abs = instance_wide_abs.merge(label_map.rename('true_label'), left_index=True, right_index=True)
    instance_wide_abs['sum_abs'] = all_df.groupby('index')['abs_weight'].sum().reindex(instance_wide_abs.index).fillna(0)
    instance_wide_abs.reset_index().to_csv(out_dir / 'lime_instance_abs_weights.csv', index=False)

    # Importância global: média dos pesos absolutos por feature
    feat_all = all_df.groupby('feature')['abs_weight'].mean().reset_index().rename(columns={'abs_weight':'mean_abs_weight'})
    feat_all = feat_all.sort_values('mean_abs_weight', ascending=False)
    feat_all.to_csv(out_dir / 'feature_importance_all.csv', index=False)

    for cls in [0,1]:
        cls_df = all_df[all_df['true_label'] == cls]
        if len(cls_df) == 0:
            continue
        feat_cls = cls_df.groupby('feature')['abs_weight'].mean().reset_index().rename(columns={'abs_weight':'mean_abs_weight'})
        feat_cls = feat_cls.sort_values('mean_abs_weight', ascending=False)
        feat_cls.to_csv(out_dir / f'feature_importance_class_{cls}.csv', index=False)

    try:
        f0 = pd.read_csv(out_dir / 'feature_importance_class_0.csv')
        f1 = pd.read_csv(out_dir / 'feature_importance_class_1.csv')
        comp = f0.merge(f1, on='feature', how='outer', suffixes=('_0','_1')).fillna(0)
        comp['diff_0_1'] = comp['mean_abs_weight_0'] - comp['mean_abs_weight_1']
        comp.to_csv(out_dir / 'comparison_class_0_vs_1.csv', index=False)
    except Exception:
        comp = None

    import matplotlib
    matplotlib.use('Agg')  # Backend sem display (para rodar em servidor)

    TOP_K = min(20, len(feat_all))
    top_features = feat_all['feature'].iloc[:TOP_K].tolist()

    plt.figure(figsize=(10, max(4, TOP_K*0.35)))
    sns.barplot(x='mean_abs_weight', y='feature', data=feat_all.head(TOP_K))
    plt.title('LIME - Importância Global (mean |coef|) - Top {}'.format(TOP_K))
    plt.tight_layout()
    plt.savefig(out_dir / 'lime_global_mean_abs_weight_top{}.png'.format(TOP_K))
    plt.close()

    for cls in [0,1]:
        csv_path = out_dir / f'feature_importance_class_{cls}.csv'
        if not csv_path.exists():
            continue
        dfc = pd.read_csv(csv_path)
        TOP_K_C = min(20, len(dfc))
        plt.figure(figsize=(10, max(4, TOP_K_C*0.35)))
        sns.barplot(x='mean_abs_weight', y='feature', data=dfc.head(TOP_K_C))
        plt.title(f'LIME - Importância Classe {cls} (mean |coef|) - Top {TOP_K_C}')
        plt.tight_layout()
        plt.savefig(out_dir / f'lime_class_{cls}_mean_abs_weight_top{TOP_K_C}.png')
        plt.close()

    df_top = all_df[all_df['feature'].isin(top_features)].copy()
    plt.figure(figsize=(12, max(4, len(top_features)*0.25)))
    sns.boxplot(x='weight', y='feature', hue='true_label', data=df_top, showfliers=False)
    plt.legend(title='true_label')
    plt.title('Distribuição dos coeficientes LIME (signed) por feature (Top {})'.format(TOP_K))
    plt.tight_layout()
    plt.savefig(out_dir / f'lime_feature_weight_distribution_top{TOP_K}.png')
    plt.close()

    pivot = all_df.pivot_table(index='index', columns='feature', values='weight', aggfunc='sum').fillna(0)
    common_cols = [c for c in top_features if c in pivot.columns]
    heatmat = pivot[common_cols].copy()
    row_labels = []
    for idx in heatmat.index:
        rec = all_df[all_df['index'] == idx]
        if len(rec) > 0:
            row_labels.append(rec['true_label'].iloc[0])
        else:
            row_labels.append(np.nan)
    heatmat['__label__'] = row_labels
    heatmat['__sum_abs__'] = heatmat[common_cols].abs().sum(axis=1)
    heatmat = heatmat.sort_values(['__label__','__sum_abs__'], ascending=[True, False])
    lbls = heatmat['__label__'].astype(int).astype(str).values
    heatmat = heatmat[common_cols]
    MAX_ROWS = 300
    if heatmat.shape[0] > MAX_ROWS:
        sampled = heatmat.head(MAX_ROWS)
        sampled_lbls = lbls[:MAX_ROWS]
    else:
        sampled = heatmat
        sampled_lbls = lbls

    plt.figure(figsize=(max(6, len(common_cols)*0.6), max(6, sampled.shape[0]*0.06)))
    sns.heatmap(sampled.values, cmap='RdBu_r', center=0, yticklabels=sampled_lbls, xticklabels=common_cols)
    plt.xlabel('Feature')
    plt.ylabel('Instância (rotulado)')
    plt.title(f'Heatmap das contribuições LIME (signed) - Top {len(common_cols)} features\nLinhas ordenadas por label e soma |contribuições|')
    plt.tight_layout()
    plt.savefig(out_dir / f'lime_heatmap_top{len(common_cols)}_rows{sampled.shape[0]}.png')
    plt.close()

    # Seleciona uma instância representativa de cada classe (maior soma de contribuições)
    rep_instances = []
    sumabs_by_idx = all_df.groupby('index')['abs_weight'].sum().reset_index().rename(columns={'abs_weight':'sum_abs'})
    label_map_simple = all_df.groupby('index')['true_label'].first().to_dict()
    for cls in [0,1]:
        cls_idxs = sumabs_by_idx[sumabs_by_idx['index'].isin([i for i,v in label_map_simple.items() if v==cls])]
        if len(cls_idxs) == 0:
            continue
        pick_idx = cls_idxs.sort_values('sum_abs', ascending=False)['index'].iloc[0]
        rep_instances.append((cls, int(pick_idx)))

    for cls, inst_idx in rep_instances:
        inst_df = all_df[all_df['index'] == inst_idx].copy()
        inst_df = inst_df.sort_values('weight', ascending=True)
        plt.figure(figsize=(8, max(3, len(inst_df)*0.4)))
        colors = inst_df['sign'].map({'pos':'#2ca02c','neg':'#d62728'})
        plt.barh(inst_df['feature'], inst_df['weight'], color=colors)
        plt.xlabel('contribuição (weight)')
        plt.title(f'LIME - Waterfall-like contributions (idx={inst_idx}, true_label={cls})')
        plt.tight_layout()
        plt.savefig(out_dir / f'lime_waterfall_idx{inst_idx}_label{cls}.png')
        plt.close()

    summary_lines = []
    summary_lines.append('# Resumo LIME - Centralizado (MLP)\n')
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
    p.add_argument('--n-jobs', type=int, default=1, help='Parallel jobs')
    p.add_argument('--random-state', type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    explain_instances(args)
