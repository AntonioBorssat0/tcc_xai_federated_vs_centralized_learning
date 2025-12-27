"""
Análise LIME para MLP Federado.

Gera explicações LIME por instância (salvas em CSV único: uma linha por instância)
e visualizações agregadas comparáveis aos outputs SHAP.

Este script espera o modelo federado em:
  flwr-mlp/models/global_model_final.joblib (ou .pt como fallback)
"""
from pathlib import Path
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from lime.lime_tabular import LimeTabularExplainer

# Adiciona paths do projeto para importar utils e federated_mlp
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "flwr-mlp"))

from utils import training_utils
from utils.shap_utils import load_shap_samples as utils_load_shap_samples, load_model_from_joblib

try:
    from federated_mlp.task import prepare_weather_data, LABEL_COL, WeatherMLP
except Exception:
    # Fallback
    prepare_weather_data = None
    LABEL_COL = 'RainTomorrow'
    WeatherMLP = None

# Carrega checkpoint .pt legado (antes de salvar em joblib)
def load_global_model_pt(model_path: str):
    ckpt = torch.load(model_path, map_location='cpu')
    if WeatherMLP is None:
        raise RuntimeError("WeatherMLP não disponível para carregar checkpoint .pt")
    base = WeatherMLP(
        input_size=ckpt['input_size'],
        hidden1=ckpt['hidden1'],
        hidden2=ckpt['hidden2'],
        hidden3=ckpt['hidden3'],
        dropout1=ckpt['dropout1'],
        dropout2=ckpt['dropout2']
    )
    base.load_state_dict(ckpt['model_state_dict'])
    base.eval()
    return base, ckpt

# Wrapper pra garantir que output sempre tenha shape correto
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        out = self.model(x)
        if out.dim() == 1:
            out = out.unsqueeze(-1)
        return out

def build_predict_proba_for_torch(model, scaler):
    """Cria função predict_proba para modelo PyTorch. Se scaler for passado, aplica antes da inferência."""
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    def predict_proba(X: np.ndarray):
        if scaler is not None:
            Xt = scaler.transform(X)
        else:
            Xt = X
        tensor = torch.tensor(Xt, dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model(tensor)
            out_cpu = out.cpu()
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

def build_predict_proba_from_joblib_artifact(artifact, scaler):
    """Cria predict_proba a partir de artefato joblib (sklearn, PyTorch ou dict com 'best_model')."""
    model = artifact
    # if dict-like
    if isinstance(artifact, dict):
        model = artifact.get('best_model', artifact)

    if hasattr(model, 'predict_proba'):
        def predict_proba(X):
            if scaler is not None:
                Xt = scaler.transform(X)
            else:
                Xt = X
            return model.predict_proba(Xt)
        return predict_proba

    if hasattr(model, 'forward') or 'torch' in str(type(model)).lower():
        return build_predict_proba_for_torch(model, scaler)

    if hasattr(model, 'predict'):
        def predict_proba(X):
            if scaler is not None:
                Xt = scaler.transform(X)
            else:
                Xt = X
            preds = model.predict(Xt)
            preds = np.asarray(preds)
            if preds.ndim == 1 and preds.dtype.kind in 'fiu':
                probs1 = preds
                probs = np.vstack([1 - probs1, probs1]).T
            elif preds.ndim == 2:
                probs = preds
            else:
                probs = np.vstack([1 - preds, preds]).T
            return probs
        return predict_proba

    raise RuntimeError("Não foi possível construir predict_proba a partir do artefato fornecido.")

def explain_instances_federated_mlp(args):
    data_path = project_root / 'datasets' / 'rain_australia' / 'weatherAUS_cleaned.csv'
    train_idx_path = project_root / 'datasets' / 'train_indices.csv'
    shap_indices_path = project_root / 'datasets' / 'shap_sample_indices.csv'

    df, train_indices, test_indices = training_utils.load_train_test_data(
        data_path, train_idx_path, project_root / 'datasets' / 'test_indices.csv'
    )

    joblib_path = project_root / 'flwr-mlp' / 'models' / 'global_model_final.joblib'
    legacy_pt = project_root / 'flwr-mlp' / 'models' / 'global_model_final.pt'

    scaler = None
    model_for_predict = None

    if joblib_path.exists():
        # Tenta carregar joblib (pode ser dict com 'best_model' e 'scaler')
        model_artifact = load_model_from_joblib(joblib_path)
        if isinstance(model_artifact, dict) and 'best_model' in model_artifact:
            base_model = model_artifact['best_model']
            scaler = model_artifact.get('scaler', None)
        else:
            base_model = model_artifact
        try:
            if hasattr(base_model, 'forward') or 'torch' in str(type(base_model)).lower():
                model_for_predict = ModelWrapper(base_model)
            else:
                model_for_predict = base_model
        except Exception:
            model_for_predict = base_model
    elif legacy_pt.exists():
        base_model, ckpt = load_global_model_pt(str(legacy_pt))
        model_for_predict = ModelWrapper(base_model)
        scaler = None
    else:
        raise FileNotFoundError(f"Modelo federado não encontrado em {joblib_path} nem {legacy_pt}")

    def preprocess_mlp(df_shap: pd.DataFrame):
        df_processed = prepare_weather_data(df_shap.copy(), use_location=False) if prepare_weather_data is not None else df_shap.copy()
        X_df = df_processed.drop(columns=[LABEL_COL])
        y_series = df_processed[LABEL_COL]
        feat_names = X_df.columns.tolist()
        X_np = X_df.values
        y_np = y_series.values
        # Se scaler disponível, usa o mesmo do treino; senão cria local
        if scaler is not None:
            X_scaled = scaler.transform(X_np)
        else:
            from sklearn.preprocessing import StandardScaler as _SS
            local_scaler = _SS()
            X_scaled = local_scaler.fit_transform(X_np)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return X_scaled, y_np, feat_names

    X_pool, y_pool, feature_names = utils_load_shap_samples(
        Path(data_path), Path(shap_indices_path), preprocess_fn=preprocess_mlp
    )

    X_train_trans, _, feature_names_train = preprocess_mlp(df.loc[train_indices])
    feature_names = feature_names_train

    # X_pool já está scaled pelo preprocess_mlp, então passa None pro scaler
    predict_proba = build_predict_proba_from_joblib_artifact(model_for_predict, None)

    explainer = LimeTabularExplainer(
        training_data=X_train_trans,
        feature_names=feature_names,
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
        group0 = np.where(y_pool == 0)[0]
        group1 = np.where(y_pool == 1)[0]
        n0 = args.sample_count // 2
        n1 = args.sample_count - n0
        chosen0 = rng.choice(group0, size=min(n0, len(group0)), replace=False) if len(group0) > 0 else np.array([], dtype=int)
        chosen1 = rng.choice(group1, size=min(n1, len(group1)), replace=False) if len(group1) > 0 else np.array([], dtype=int)
        chosen_local = np.concatenate([chosen0, chosen1])

    shap_df = pd.read_csv(shap_indices_path)
    pool_indices = shap_df['index'].values
    chosen = pool_indices[chosen_local]

    out_dir = project_root / 'xAI' / 'lime_results_federated' / 'mlp'
    out_dir.mkdir(parents=True, exist_ok=True)

    per_instance_records = []

    print(f"Gerando explicações LIME para {len(chosen_local)} instâncias (MLP Federado)...")
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
        print("Nenhuma explicação foi gerada. Saindo.")
        return

    all_df = pd.concat(per_instance_records, ignore_index=True)

    # Cria CSV com uma linha por instância (colunas = features com pesos signed)
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
    matplotlib.use('Agg')

    TOP_K = min(20, len(feat_all))
    top_features = feat_all['feature'].iloc[:TOP_K].tolist()

    plt.figure(figsize=(10, max(4, TOP_K*0.35)))
    sns.barplot(x='mean_abs_weight', y='feature', data=feat_all.head(TOP_K))
    plt.title('LIME - Importância Global (mean |coef|) - MLP Federado - Top {}'.format(TOP_K))
    plt.tight_layout()
    plt.savefig(out_dir / f'lime_fed_mlp_global_mean_abs_weight_top{TOP_K}.png')
    plt.close()

    for cls in [0,1]:
        csv_path = out_dir / f'feature_importance_class_{cls}.csv'
        if not Path(csv_path).exists():
            continue
        dfc = pd.read_csv(csv_path)
        TOP_K_C = min(20, len(dfc))
        plt.figure(figsize=(10, max(4, TOP_K_C*0.35)))
        sns.barplot(x='mean_abs_weight', y='feature', data=dfc.head(TOP_K_C))
        plt.title(f'LIME Federado MLP - Importância Classe {cls} (mean |coef|) - Top {TOP_K_C}')
        plt.tight_layout()
        plt.savefig(out_dir / f'lime_fed_mlp_class_{cls}_mean_abs_weight_top{TOP_K_C}.png')
        plt.close()

    df_top = all_df[all_df['feature'].isin(top_features)].copy()
    plt.figure(figsize=(12, max(4, len(top_features)*0.25)))
    sns.boxplot(x='weight', y='feature', hue='true_label', data=df_top, showfliers=False)
    plt.legend(title='true_label')
    plt.title('Distribuição dos coeficientes LIME (signed) por feature (Top {}) - MLP Federado'.format(TOP_K))
    plt.tight_layout()
    plt.savefig(out_dir / f'lime_fed_mlp_feature_weight_distribution_top{TOP_K}.png')
    plt.close()

    pivot = all_df.pivot_table(index='index', columns='feature', values='weight', aggfunc='sum').fillna(0)
    common_cols = [c for c in top_features if c in pivot.columns]
    heatmat = pivot[common_cols].copy()
    row_labels = []
    for idx in heatmat.index:
        rec = all_df[all_df['index'] == idx]
        row_labels.append(rec['true_label'].iloc[0] if len(rec)>0 else np.nan)
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
    plt.title(f'Heatmap das contribuições LIME (signed) - MLP Federado - Top {len(common_cols)} features')
    plt.tight_layout()
    plt.savefig(out_dir / f'lime_fed_mlp_heatmap_top{len(common_cols)}_rows{sampled.shape[0]}.png')
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
        inst_df = all_df[all_df['index'] == inst_idx].copy().sort_values('weight', ascending=True)
        plt.figure(figsize=(8, max(3, len(inst_df)*0.4)))
        colors = inst_df['sign'].map({'pos':'#2ca02c','neg':'#d62728'})
        plt.barh(inst_df['feature'], inst_df['weight'], color=colors)
        plt.xlabel('contribuição (weight)')
        plt.title(f'LIME - Waterfall-like contributions (idx={inst_idx}, true_label={cls}) - MLP Federado')
        plt.tight_layout()
        plt.savefig(out_dir / f'lime_fed_mlp_waterfall_idx{inst_idx}_label{cls}.png')
        plt.close()

    summary_lines = []
    summary_lines.append('# Resumo LIME - Federado (MLP)\n')
    summary_lines.append(f'Instâncias explicadas (sample_count): {len(chosen_local)}\n')
    summary_lines.append(f'Top {TOP_K} features (global, mean |weight|):\n')
    for _, row in feat_all.head(TOP_K).iterrows():
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
    explain_instances_federated_mlp(args)
