"""Gera tabelas auxiliares (CSV) para o artigo.

Saídas em xAI/:
- shap_feature_rankings_19_long.csv
- shap_feature_rankings_19_wide.csv
- kendall_tau_shap_rankings.csv
- kendall_tau_shap_rankings_pvalues.csv
- kendall_tau_distance_feature_rankings_19.csv
- kendall_tau_distance_shap_rankings.csv
- kendall_tau_distance_lime_rankings.csv
- lime_local_fidelity_comparison.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import kendalltau
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "SciPy is required for Kendall tau computation. "
        "It is declared in the root pyproject.toml dependencies."
    ) from e


XAI_DIR = Path(__file__).resolve().parent


SHAP_PATHS: Dict[str, Path] = {
    "MLP Centralizado": XAI_DIR / "shap_results_centralized" / "mlp" / "feature_importance_all.csv",
    "XGBoost Centralizado": XAI_DIR / "shap_results_centralized" / "xgboost" / "feature_importance_all.csv",
    "MLP Federado (FedAvg)": XAI_DIR / "shap_results_federated" / "mlp" / "feature_importance_all.csv",
    "XGBoost Fed. (Bagging)": XAI_DIR
    / "shap_results_federated"
    / "xgboost"
    / "bagging_strategy"
    / "feature_importance_all.csv",
    "XGBoost Fed. (Cyclic)": XAI_DIR
    / "shap_results_federated"
    / "xgboost"
    / "cyclic_strategy"
    / "feature_importance_all.csv",
}


LIME_IMPORTANCE_PATHS: Dict[str, Path] = {
    "MLP Centralizado": XAI_DIR / "lime_results_centralized" / "mlp" / "feature_importance_all.csv",
    "XGBoost Centralizado": XAI_DIR / "lime_results_centralized" / "xgboost" / "feature_importance_all.csv",
    "MLP Federado (FedAvg)": XAI_DIR / "lime_results_federated" / "mlp" / "feature_importance_all.csv",
    "XGBoost Fed. (Bagging)": XAI_DIR
    / "lime_results_federated"
    / "xgboost"
    / "bagging_strategy"
    / "feature_importance_all.csv",
    "XGBoost Fed. (Cyclic)": XAI_DIR
    / "lime_results_federated"
    / "xgboost"
    / "cyclic_strategy"
    / "feature_importance_all.csv",
}


LIME_FIDELITY_PATHS: Dict[str, Path] = {
    "MLP Centralizado": XAI_DIR / "lime_results_centralized" / "mlp" / "lime_instance_fidelity.csv",
    "XGBoost Centralizado": XAI_DIR / "lime_results_centralized" / "xgboost" / "lime_instance_fidelity.csv",
    "MLP Federado (FedAvg)": XAI_DIR / "lime_results_federated" / "mlp" / "lime_instance_fidelity.csv",
    "XGBoost Fed. (Bagging)": XAI_DIR
    / "lime_results_federated"
    / "xgboost"
    / "bagging_strategy"
    / "lime_instance_fidelity.csv",
    "XGBoost Fed. (Cyclic)": XAI_DIR
    / "lime_results_federated"
    / "xgboost"
    / "cyclic_strategy"
    / "lime_instance_fidelity.csv",
}


MODEL_ORDER: List[str] = [
    "XGBoost Centralizado",
    "MLP Centralizado",
    "MLP Federado (FedAvg)",
    "XGBoost Fed. (Bagging)",
    "XGBoost Fed. (Cyclic)",
]


def _read_shap_importance_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"SHAP CSV not found: {path}")

    df = pd.read_csv(path)

    # Normaliza nomes das colunas
    if "Feature" not in df.columns:
        if "feature" in df.columns:
            df = df.rename(columns={"feature": "Feature"})
        else:
            raise ValueError(f"Expected 'Feature' column in {path}, got: {list(df.columns)}")

    if "Mean_Abs_SHAP" in df.columns:
        df = df.rename(columns={"Mean_Abs_SHAP": "Importance"})
    elif "mean_abs_shap" in df.columns:
        df = df.rename(columns={"mean_abs_shap": "Importance"})
    elif "Importance" not in df.columns:
        if "importance" in df.columns:
            df = df.rename(columns={"importance": "Importance"})
        else:
            raise ValueError(
                f"Expected one of ['Mean_Abs_SHAP','mean_abs_shap','Importance','importance'] in {path}, "
                f"got: {list(df.columns)}"
            )

    df = df[["Feature", "Importance"]].copy()
    df["Importance"] = pd.to_numeric(df["Importance"], errors="coerce")

    if df["Importance"].isna().any():
        bad = df[df["Importance"].isna()]["Feature"].tolist()
        raise ValueError(f"Non-numeric Importance values in {path} for features: {bad}")

    return df


def _read_lime_importance_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"LIME CSV not found: {path}")

    df = pd.read_csv(path)

    if "Feature" not in df.columns:
        if "feature" in df.columns:
            df = df.rename(columns={"feature": "Feature"})
        else:
            raise ValueError(f"Expected 'feature' column in {path}, got: {list(df.columns)}")

    if "Importance" not in df.columns:
        if "mean_abs_weight" in df.columns:
            df = df.rename(columns={"mean_abs_weight": "Importance"})
        elif "Mean_Abs_Weight" in df.columns:
            df = df.rename(columns={"Mean_Abs_Weight": "Importance"})
        else:
            raise ValueError(
                f"Expected one of ['mean_abs_weight','Mean_Abs_Weight','Importance'] in {path}, got: {list(df.columns)}"
            )

    df = df[["Feature", "Importance"]].copy()
    df["Importance"] = pd.to_numeric(df["Importance"], errors="coerce")
    if df["Importance"].isna().any():
        bad = df[df["Importance"].isna()]["Feature"].tolist()
        raise ValueError(f"Non-numeric Importance values in {path} for features: {bad}")

    return df


def _compute_ranks_strict_and_tau(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Retorna (strict_rank, tau_rank) indexados por Feature."""

    s = df.set_index("Feature")["Importance"].copy()

    # tau_rank: mantém empates
    tau_rank = s.rank(method="average", ascending=False)

    # strict_rank: desempata de forma determinística
    strict_df = (
        s.reset_index()
        .rename(columns={"Importance": "importance"})
        .sort_values(["importance", "Feature"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    strict_df["strict_rank"] = np.arange(1, len(strict_df) + 1, dtype=int)
    strict_rank = strict_df.set_index("Feature")["strict_rank"].astype(int)

    return strict_rank, tau_rank


def _kendall_distance_normalized(order_a: List[str], order_b: List[str]) -> float:
    """Kendall tau distance normalizada: K in [0,1].

    Para permutações sem empates: K = D / (n choose 2), onde D é o número de pares discordantes.
    """

    if len(order_a) != len(order_b):
        raise ValueError("Orders must have same length")

    n = len(order_a)
    if n < 2:
        return 0.0

    idx_b = {feat: i for i, feat in enumerate(order_b)}
    if len(idx_b) != n:
        raise ValueError("Order has duplicate features")

    discordant = 0
    for i in range(n):
        fi = order_a[i]
        for j in range(i + 1, n):
            fj = order_a[j]
            if idx_b[fi] > idx_b[fj]:
                discordant += 1

    total_pairs = n * (n - 1) / 2
    return float(discordant / total_pairs)


def _validate_19_same_features(model_to_df: Dict[str, pd.DataFrame], label: str) -> List[str]:
    feature_sets = {name: set(df["Feature"].tolist()) for name, df in model_to_df.items()}
    expected = next(iter(feature_sets.values()))
    problems: List[str] = []

    for name, feat_set in feature_sets.items():
        if len(feat_set) != 19:
            problems.append(f"{label} - {name}: expected 19 features, found {len(feat_set)}")
        if feat_set != expected:
            missing = sorted(list(expected - feat_set))
            extra = sorted(list(feat_set - expected))
            problems.append(f"{label} - {name}: feature set mismatch (missing={missing}, extra={extra})")

    return problems


def _orders_from_importances(model_to_df: Dict[str, pd.DataFrame]) -> Tuple[List[str], Dict[str, List[str]]]:
    feature_set = set(next(iter(model_to_df.values()))["Feature"].tolist())
    features_sorted = sorted(feature_set)
    orders: Dict[str, List[str]] = {}
    for model in MODEL_ORDER:
        strict_rank, _ = _compute_ranks_strict_and_tau(model_to_df[model])
        # strict_rank index = Feature, values 1..n
        orders[model] = strict_rank.sort_values().index.tolist()
    return features_sorted, orders


def generate_kendall_tau_distance_tables(out_dir: Path) -> Tuple[Path, Path, Path]:
    """Gera distâncias K (0..1) para rankings de importância (19 features)."""

    out_dir.mkdir(parents=True, exist_ok=True)

    shap_data: Dict[str, pd.DataFrame] = {
        model: _read_shap_importance_csv(SHAP_PATHS[model]) for model in MODEL_ORDER
    }
    lime_data: Dict[str, pd.DataFrame] = {
        model: _read_lime_importance_csv(LIME_IMPORTANCE_PATHS[model]) for model in MODEL_ORDER
    }

    problems = _validate_19_same_features(shap_data, label="SHAP")
    problems += _validate_19_same_features(lime_data, label="LIME")
    if problems:
        raise ValueError("Feature validation failed: " + "; ".join(problems))

    _, shap_orders = _orders_from_importances(shap_data)
    _, lime_orders = _orders_from_importances(lime_data)

    shap_k = pd.DataFrame(index=MODEL_ORDER, columns=MODEL_ORDER, dtype=float)
    lime_k = pd.DataFrame(index=MODEL_ORDER, columns=MODEL_ORDER, dtype=float)

    for i, mi in enumerate(MODEL_ORDER):
        for j, mj in enumerate(MODEL_ORDER):
            if i == j:
                shap_k.loc[mi, mj] = 0.0
                lime_k.loc[mi, mj] = 0.0
                continue
            shap_k.loc[mi, mj] = _kendall_distance_normalized(shap_orders[mi], shap_orders[mj])
            lime_k.loc[mi, mj] = _kendall_distance_normalized(lime_orders[mi], lime_orders[mj])

    shap_k_csv = out_dir / "kendall_tau_distance_shap_rankings.csv"
    lime_k_csv = out_dir / "kendall_tau_distance_lime_rankings.csv"
    shap_k.to_csv(shap_k_csv, index=True)
    lime_k.to_csv(lime_k_csv, index=True)

    rows: List[dict] = []

    def add_cross(label: str, a: str, b: str) -> None:
        rows.append(
            {
                "Comparison": label,
                "K_SHAP": _kendall_distance_normalized(shap_orders[a], shap_orders[b]),
                "K_LIME": _kendall_distance_normalized(lime_orders[a], lime_orders[b]),
                "K_SHAP_vs_LIME": np.nan,
            }
        )

    def add_within(model: str, label: str) -> None:
        rows.append(
            {
                "Comparison": label,
                "K_SHAP": np.nan,
                "K_LIME": np.nan,
                "K_SHAP_vs_LIME": _kendall_distance_normalized(shap_orders[model], lime_orders[model]),
            }
        )

    add_cross(
        "Centralized MLP vs. Federated MLP",
        "MLP Centralizado",
        "MLP Federado (FedAvg)",
    )
    add_cross(
        "Centralized XGBoost vs. Fed. XGBoost (Bagging)",
        "XGBoost Centralizado",
        "XGBoost Fed. (Bagging)",
    )
    add_cross(
        "Centralized XGBoost vs. Fed. XGBoost (Cyclic)",
        "XGBoost Centralizado",
        "XGBoost Fed. (Cyclic)",
    )

    add_within("MLP Centralizado", "SHAP vs. LIME (Centralized MLP)")
    add_within("MLP Federado (FedAvg)", "SHAP vs. LIME (Federated MLP)")
    add_within("XGBoost Centralizado", "SHAP vs. LIME (Centralized XGBoost)")
    add_within("XGBoost Fed. (Bagging)", "SHAP vs. LIME (Fed. XGBoost Bagging)")
    add_within("XGBoost Fed. (Cyclic)", "SHAP vs. LIME (Fed. XGBoost Cyclic)")

    comparisons_csv = out_dir / "kendall_tau_distance_feature_rankings_19.csv"
    pd.DataFrame(rows).to_csv(comparisons_csv, index=False)

    return comparisons_csv, shap_k_csv, lime_k_csv


def generate_shap_rankings_and_kendall(out_dir: Path) -> Tuple[Path, Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    shap_data: Dict[str, pd.DataFrame] = {
        model: _read_shap_importance_csv(SHAP_PATHS[model]) for model in MODEL_ORDER
    }

    # Valida features (19 e conjunto idêntico entre modelos)
    feature_sets = {name: set(df["Feature"].tolist()) for name, df in shap_data.items()}
    expected = next(iter(feature_sets.values()))
    problems: List[str] = []

    for name, feat_set in feature_sets.items():
        if len(feat_set) != 19:
            problems.append(f"{name}: expected 19 features, found {len(feat_set)}")
        if feat_set != expected:
            missing = sorted(list(expected - feat_set))
            extra = sorted(list(feat_set - expected))
            problems.append(f"{name}: feature set mismatch (missing={missing}, extra={extra})")

    if problems:
        raise ValueError("SHAP feature validation failed: " + "; ".join(problems))

    features_sorted = sorted(expected)

    # Monta rankings em formato long e wide
    long_rows: List[dict] = []
    wide = pd.DataFrame({"Feature": features_sorted})
    tau_ranks: Dict[str, pd.Series] = {}

    for model in MODEL_ORDER:
        df = shap_data[model]
        strict_rank, tau_rank = _compute_ranks_strict_and_tau(df)
        tau_ranks[model] = tau_rank

        importance = df.set_index("Feature")["Importance"]

        for feat in strict_rank.sort_values().index.tolist():
            long_rows.append(
                {
                    "Modelo": model,
                    "Rank": int(strict_rank.loc[feat]),
                    "Feature": feat,
                    "Importance": float(importance.loc[feat]),
                }
            )

        wide[f"Rank - {model}"] = wide["Feature"].map(strict_rank).astype(int)
        wide[f"Importance - {model}"] = wide["Feature"].map(importance).astype(float)

    long_df = (
        pd.DataFrame(long_rows)
        .sort_values(["Modelo", "Rank"], ascending=[True, True])
        .reset_index(drop=True)
    )

    rankings_long_csv = out_dir / "shap_feature_rankings_19_long.csv"
    rankings_wide_csv = out_dir / "shap_feature_rankings_19_wide.csv"
    long_df.to_csv(rankings_long_csv, index=False)
    wide.to_csv(rankings_wide_csv, index=False)

    # Matrizes de Kendall tau
    tau_mat = pd.DataFrame(index=MODEL_ORDER, columns=MODEL_ORDER, dtype=float)
    p_mat = pd.DataFrame(index=MODEL_ORDER, columns=MODEL_ORDER, dtype=float)

    for i, mi in enumerate(MODEL_ORDER):
        for j, mj in enumerate(MODEL_ORDER):
            if i == j:
                tau_mat.loc[mi, mj] = 1.0
                p_mat.loc[mi, mj] = 0.0
                continue

            a = tau_ranks[mi].reindex(features_sorted).to_numpy(dtype=float)
            b = tau_ranks[mj].reindex(features_sorted).to_numpy(dtype=float)
            res = kendalltau(a, b, nan_policy="raise")
            tau_mat.loc[mi, mj] = float(res.statistic)
            p_mat.loc[mi, mj] = float(res.pvalue)

    kendall_tau_csv = out_dir / "kendall_tau_shap_rankings.csv"
    kendall_p_csv = out_dir / "kendall_tau_shap_rankings_pvalues.csv"
    tau_mat.to_csv(kendall_tau_csv, index=True)
    p_mat.to_csv(kendall_p_csv, index=True)

    return rankings_long_csv, rankings_wide_csv, kendall_tau_csv, kendall_p_csv


def generate_lime_fidelity_summary(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    for model in MODEL_ORDER:
        path = LIME_FIDELITY_PATHS[model]
        if not path.exists():
            rows.append(
                {
                    "Modelo": model,
                    "mean_r2": np.nan,
                    "std_r2": np.nan,
                    "n": 0,
                    "path": str(path.relative_to(XAI_DIR)),
                }
            )
            continue

        df = pd.read_csv(path)
        if "local_fidelity_r2" not in df.columns:
            raise ValueError(f"Expected 'local_fidelity_r2' in {path}, got: {list(df.columns)}")

        r2 = pd.to_numeric(df["local_fidelity_r2"], errors="coerce").dropna()
        rows.append(
            {
                "Modelo": model,
                "mean_r2": float(r2.mean()) if len(r2) else np.nan,
                "std_r2": float(r2.std(ddof=1)) if len(r2) > 1 else 0.0,
                "n": int(len(r2)),
                "path": str(path.relative_to(XAI_DIR)),
            }
        )

    out_csv = out_dir / "lime_local_fidelity_comparison.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def main() -> None:
    out_dir = XAI_DIR

    print("Generating SHAP rankings + Kendall tau…")
    rankings_long_csv, rankings_wide_csv, kendall_tau_csv, kendall_p_csv = generate_shap_rankings_and_kendall(out_dir)

    print("Generating LIME local fidelity summary…")
    lime_csv = generate_lime_fidelity_summary(out_dir)

    print("Generating Kendall tau distance K tables (SHAP/LIME)…")
    comparisons_csv, shap_k_csv, lime_k_csv = generate_kendall_tau_distance_tables(out_dir)

    print("\nDone. Files written:")
    print(f"- {rankings_long_csv}")
    print(f"- {rankings_wide_csv}")
    print(f"- {kendall_tau_csv}")
    print(f"- {kendall_p_csv}")
    print(f"- {lime_csv}")
    print(f"- {comparisons_csv}")
    print(f"- {shap_k_csv}")
    print(f"- {lime_k_csv}")


if __name__ == "__main__":
    main()
