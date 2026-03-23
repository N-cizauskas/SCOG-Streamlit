"""
Streamlit UI for CTGAN Synthetic Data Generation

"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import base64
from pathlib import Path
from datetime import datetime
from io import BytesIO
import textwrap
from scipy.stats import norm
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from matplotlib.backends.backend_pdf import PdfPages

try:
    from PIL import Image
except ImportError:
    Image = None

# add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import DataLoader
from CTGAN_dec_adjustable import CustomCTGAN, CorrelationAwareCTGAN, OrdinalCTGAN
from evaluation import EvaluationMetrics

# page icon (prefer local head image)
page_icon = "✨"
page_icon_candidates = [
    Path(__file__).parent / "github_site" / "SCOG_pics" / "head_detail.png",
    Path(__file__).parent / "SCOG_pics" / "head_detail.png",
    Path(__file__).parent / "github_site" / "SCOG_pics" / "head.png",
    Path(__file__).parent / "SCOG_pics" / "head.png"
]
for icon_path in page_icon_candidates:
    if icon_path.exists():
        if Image is not None:
            try:
                icon_img = Image.open(icon_path).convert("RGBA")
                bbox = icon_img.getbbox()
                if bbox is not None:
                    icon_img = icon_img.crop(bbox)
                icon_img = icon_img.resize((512, 512), Image.Resampling.LANCZOS)
                page_icon = icon_img
            except Exception:
                page_icon = str(icon_path)
        else:
            page_icon = str(icon_path)
        break

# configure page
st.set_page_config(
    page_title="SCOG: Synthetic Control Generator",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

CB_BLUE = "#0072B2"
CB_BROWN = "#8C510A"


def _load_streamlit_bullet_icon_data_uri() -> str:
    bullet_icon_path = Path(__file__).parent / "github_site" / "SCOG_pics" / "bulletpoints.png"
    if not bullet_icon_path.exists():
        return ""
    with open(bullet_icon_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


BULLET_ICON_DATA_URI = _load_streamlit_bullet_icon_data_uri()


# custom markdown
custom_css = """
    <style>
    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5em;
    }
    .section-header {
        font-size: 1.5em;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 1em;
        margin-bottom: 0.5em;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 0.5em 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 0.5em 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 0.5em 0;
    }
    .katex-display {
        font-size: 0.92em;
        white-space: nowrap;
        overflow-x: auto;
        overflow-y: hidden;
    }
    .math-lay-summary {
        color: #1f77b4;
        margin-top: -0.25rem;
        margin-bottom: 0.6rem;
        font-size: 0.9rem;
    }
    .sidebar-blink-wrap {
        text-align: center;
        margin-bottom: -0.22rem;
    }
    .sidebar-blink-wrap img {
        width: 220px;
        max-width: 100%;
        height: auto;
    }
    .sidebar-logo-wrap {
        text-align: center;
        margin-top: -1.0rem;
        margin-bottom: 0.35rem;
    }
    .sidebar-logo-wrap img {
        width: 280px;
        max-width: 100%;
        height: auto;
    }
    .stApp ul {
        list-style: none;
        padding-left: 0;
    }
    .stApp ul li {
        position: relative;
        padding-left: 1.78rem;
    }
    .stApp ul li::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0.18rem;
        width: 1.15rem;
        height: 1.15rem;
        background-image: url('__BULLET_ICON__');
        background-repeat: no-repeat;
        background-position: center;
        background-size: contain;
    }
    </style>
"""
st.markdown(custom_css.replace("__BULLET_ICON__", BULLET_ICON_DATA_URI), unsafe_allow_html=True)

# start session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'continuous_cols' not in st.session_state:
    st.session_state.continuous_cols = []
if 'categorical_cols' not in st.session_state:
    st.session_state.categorical_cols = []
if 'binary_cols' not in st.session_state:
    st.session_state.binary_cols = []
if 'ordinal_cols' not in st.session_state:
    st.session_state.ordinal_cols = []
if 'ordinal_orders' not in st.session_state:
    st.session_state.ordinal_orders = {}
if 'condition_col' not in st.session_state:
    st.session_state.condition_col = None
if 'outlier_enabled' not in st.session_state:
    st.session_state.outlier_enabled = False
if 'outlier_columns' not in st.session_state:
    st.session_state.outlier_columns = []
if 'outlier_iqr_multiplier' not in st.session_state:
    st.session_state.outlier_iqr_multiplier = 1.5
if 'outlier_combination_rule' not in st.session_state:
    st.session_state.outlier_combination_rule = 'OR'
if 'missing_data_action' not in st.session_state:
    st.session_state.missing_data_action = 'drop'
if 'missing_num_impute' not in st.session_state:
    st.session_state.missing_num_impute = 'median'
if 'missing_cat_impute' not in st.session_state:
    st.session_state.missing_cat_impute = 'mode'
if 'missing_cat_fill_value' not in st.session_state:
    st.session_state.missing_cat_fill_value = 'Unknown'
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'synthetic_df' not in st.session_state:
    st.session_state.synthetic_df = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'use_correlation_model' not in st.session_state:
    st.session_state.use_correlation_model = False
if 'correlation_baseline' not in st.session_state:
    st.session_state.correlation_baseline = None
if 'comparison_results_df' not in st.session_state:
    st.session_state.comparison_results_df = None
if 'comparison_synth_data' not in st.session_state:
    st.session_state.comparison_synth_data = {}
if 'comparison_selected_metrics' not in st.session_state:
    st.session_state.comparison_selected_metrics = []


def compute_correlation_report(real_df: pd.DataFrame, synth_df: pd.DataFrame | None = None):
    numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return None

    real_num = real_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    real_corr = real_num.corr(method='spearman')
    real_strength = real_corr.abs().where(~np.eye(len(real_corr), dtype=bool))
    real_mean_abs = float(np.nanmean(real_strength.values))

    report = {
        'numeric_cols': numeric_cols,
        'real_corr': real_corr,
        'real_mean_abs_corr': real_mean_abs,
    }

    if synth_df is not None:
        common_numeric_cols = [c for c in numeric_cols if c in synth_df.columns]
        report['matched_numeric_cols'] = common_numeric_cols
        if len(common_numeric_cols) < 2:
            return report

        real_common = real_df[common_numeric_cols].apply(pd.to_numeric, errors='coerce')
        synth_common = synth_df[common_numeric_cols].apply(pd.to_numeric, errors='coerce')

        valid_cols = [
            c for c in common_numeric_cols
            if real_common[c].notna().sum() > 1 and synth_common[c].notna().sum() > 1
        ]
        report['valid_numeric_cols'] = valid_cols
        if len(valid_cols) < 2:
            return report

        synth_num = synth_common[valid_cols]
        real_for_diff = real_common[valid_cols]

        synth_corr = synth_num.corr(method='spearman')
        real_corr_for_diff = real_for_diff.corr(method='spearman')
        diff = (synth_corr - real_corr_for_diff).abs()
        diff_masked = diff.where(~np.eye(len(diff), dtype=bool))
        finite_vals = diff_masked.values[np.isfinite(diff_masked.values)]
        max_abs_diff = float(np.max(finite_vals)) if finite_vals.size > 0 else float('nan')
        report.update({
            'synth_corr': synth_corr,
            'abs_diff': diff,
            'mean_abs_diff': float(np.nanmean(diff_masked.values)),
            'max_abs_diff': max_abs_diff,
        })

    return report


def _approx_wasserstein_distance_1d(real_values: np.ndarray, synth_values: np.ndarray) -> float:
    real_values = np.asarray(real_values, dtype=float)
    synth_values = np.asarray(synth_values, dtype=float)
    real_values = real_values[np.isfinite(real_values)]
    synth_values = synth_values[np.isfinite(synth_values)]
    if real_values.size == 0 or synth_values.size == 0:
        return float('nan')

    n_points = int(max(real_values.size, synth_values.size, 32))
    q = np.linspace(0.0, 1.0, n_points)
    real_q = np.quantile(real_values, q)
    synth_q = np.quantile(synth_values, q)
    return float(np.mean(np.abs(real_q - synth_q)))


def compute_dimension_wise_distance(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    continuous_cols: list[str],
    categorical_cols: list[str],
):
    rows = []
    eps = 1e-8

    for col in continuous_cols:
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        real_vals = pd.to_numeric(real_df[col], errors='coerce').values
        synth_vals = pd.to_numeric(synth_df[col], errors='coerce').values
        wd = _approx_wasserstein_distance_1d(real_vals, synth_vals)
        if np.isnan(wd):
            continue

        real_finite = real_vals[np.isfinite(real_vals)]
        if real_finite.size == 0:
            continue
        iqr = float(np.nanpercentile(real_finite, 75) - np.nanpercentile(real_finite, 25))
        scale = iqr if iqr > eps else float(np.nanstd(real_finite))
        scale = scale if scale > eps else 1.0

        wd_norm = wd / scale
        score = wd_norm / (1.0 + wd_norm)
        rows.append({
            'column': col,
            'type': 'continuous',
            'metric': 'wasserstein',
            'raw_value': wd,
            'normalized_score': score,
        })

    for col in categorical_cols:
        if col not in real_df.columns or col not in synth_df.columns:
            continue

        real_counts = real_df[col].astype(str).value_counts(normalize=True)
        synth_counts = synth_df[col].astype(str).value_counts(normalize=True)
        all_levels = sorted(set(real_counts.index).union(set(synth_counts.index)))
        if len(all_levels) == 0:
            continue

        real_dist = real_counts.reindex(all_levels, fill_value=0.0)
        synth_dist = synth_counts.reindex(all_levels, fill_value=0.0)
        abs_diffs = (real_dist - synth_dist).abs()
        tv_distance = float(0.5 * abs_diffs.sum())

        rows.append({
            'column': col,
            'type': 'categorical',
            'metric': 'abs_prevalence_diff',
            'raw_value': tv_distance,
            'normalized_score': tv_distance,
        })

    if not rows:
        return None

    details_df = pd.DataFrame(rows)
    raw_sum = float(details_df['normalized_score'].sum())
    normalized_sum = raw_sum / max(1, len(details_df))
    l1_sum = raw_sum
    l1_normalized = normalized_sum

    return {
        'details': details_df.sort_values(['type', 'column']).reset_index(drop=True),
        'raw_sum': raw_sum,
        'normalized_sum': normalized_sum,
        'l1_sum': l1_sum,
        'l1_normalized': l1_normalized,
        'dimension_wise_distance': normalized_sum,
        'n_columns': int(len(details_df)),
    }


def compute_iqr_outlier_mask(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    values = pd.to_numeric(series, errors='coerce')
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return pd.Series(False, index=series.index)
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (values < lower) | (values > upper)


def infer_column_groups_for_comparison(df: pd.DataFrame):
    continuous_cols = []
    categorical_cols = []
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(df[col]) and nunique > 10:
            continuous_cols.append(col)
        else:
            categorical_cols.append(col)
    return continuous_cols, categorical_cols


def _align_real_synth_for_metrics(real_df: pd.DataFrame, synth_df: pd.DataFrame):
    common_cols = [c for c in real_df.columns if c in synth_df.columns]
    if len(common_cols) == 0:
        return None, None
    real_aligned = real_df[common_cols].copy()
    synth_aligned = synth_df[common_cols].copy()
    return real_aligned, synth_aligned


def generate_random_row_sample(df: pd.DataFrame, n_rows: int, random_state: int = 42) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()
    return df.sample(n=n_rows, replace=(len(df) < n_rows), random_state=random_state).reset_index(drop=True)


def generate_independent_column_sample(df: pd.DataFrame, n_rows: int, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    output = {}
    for col in df.columns:
        values = df[col].values
        if len(values) == 0:
            output[col] = np.array([np.nan] * n_rows)
            continue
        idx = rng.integers(0, len(values), size=n_rows)
        output[col] = values[idx]
    return pd.DataFrame(output)


def generate_gaussian_copula_sample(df: pd.DataFrame, n_rows: int, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    synth = pd.DataFrame(index=range(n_rows))

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    # categorical features: frequency sampling
    for col in categorical_cols:
        probs = df[col].astype(str).value_counts(normalize=True)
        if len(probs) == 0:
            synth[col] = [np.nan] * n_rows
            continue
        synth[col] = rng.choice(probs.index.to_numpy(), size=n_rows, p=probs.values)

    # numeric features: gaussian copula via rank transform
    if len(numeric_cols) == 0:
        return synth[df.columns]
    if len(numeric_cols) == 1:
        col = numeric_cols[0]
        observed = pd.to_numeric(df[col], errors='coerce').dropna().values
        if len(observed) == 0:
            synth[col] = np.nan
        else:
            q = rng.uniform(0.0, 1.0, size=n_rows)
            synth[col] = np.quantile(observed, q)
        return synth[df.columns]

    numeric_df = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    numeric_filled = numeric_df.copy()
    for col in numeric_cols:
        median_val = numeric_filled[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        numeric_filled[col] = numeric_filled[col].fillna(median_val)

    ranks = numeric_filled.rank(method='average', pct=True).clip(1e-4, 1 - 1e-4)
    gaussian_scores = pd.DataFrame(norm.ppf(ranks.values), columns=numeric_cols)

    corr = gaussian_scores.corr().values
    corr = np.nan_to_num(corr, nan=0.0)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    corr += np.eye(corr.shape[0]) * 1e-6

    z = rng.multivariate_normal(
        mean=np.zeros(len(numeric_cols)),
        cov=corr,
        size=n_rows,
        check_valid='warn',
    )
    u = norm.cdf(z)

    for j, col in enumerate(numeric_cols):
        observed = numeric_filled[col].values
        synth[col] = np.quantile(observed, u[:, j])

    return synth[df.columns]


def generate_cart_leaf_bootstrap(df: pd.DataFrame, n_rows: int, random_state: int = 42) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()

    rng = np.random.default_rng(random_state)

    # choose a target for CART partitioning
    categorical_target_candidates = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique(dropna=True) <= 10]
    if categorical_target_candidates:
        target_col = categorical_target_candidates[0]
        y = df[target_col].astype(str).fillna('__NA__')
    else:
        target_col = df.columns[0]
        numeric_target = pd.to_numeric(df[target_col], errors='coerce')
        if numeric_target.notna().sum() > 0:
            y = pd.qcut(numeric_target.fillna(numeric_target.median()), q=min(5, max(2, numeric_target.nunique())), duplicates='drop').astype(str)
        else:
            y = pd.Series(['bin_0'] * len(df), index=df.index)

    feature_df = pd.get_dummies(df.drop(columns=[target_col]), dummy_na=True)
    if feature_df.shape[1] == 0:
        return generate_random_row_sample(df, n_rows, random_state=random_state)

    min_leaf = max(5, len(df) // 50)
    tree = DecisionTreeClassifier(max_depth=6, min_samples_leaf=min_leaf, random_state=random_state)
    tree.fit(feature_df, y)

    leaves = tree.apply(feature_df)
    leaf_series = pd.Series(leaves, index=df.index)
    leaf_counts = leaf_series.value_counts().sort_index()
    leaf_ids = leaf_counts.index.to_numpy()
    leaf_probs = (leaf_counts / leaf_counts.sum()).values

    chosen_leaves = rng.choice(leaf_ids, size=n_rows, p=leaf_probs)
    sampled_indices = []
    for leaf_id in chosen_leaves:
        pool = leaf_series[leaf_series == leaf_id].index.to_numpy()
        sampled_indices.append(int(rng.choice(pool)))

    return df.loc[sampled_indices].reset_index(drop=True)


def generate_bayesian_bootstrap_sample(df: pd.DataFrame, n_rows: int, random_state: int = 42) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()
    rng = np.random.default_rng(random_state)
    probs = rng.dirichlet(np.ones(len(df)))
    idx = rng.choice(np.arange(len(df)), size=n_rows, replace=True, p=probs)
    return df.iloc[idx].reset_index(drop=True)


def generate_kmeans_cluster_bootstrap(df: pd.DataFrame, n_rows: int, random_state: int = 42) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0 or len(df) < 8:
        return generate_bayesian_bootstrap_sample(df, n_rows, random_state=random_state)

    numeric_df = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    for col in numeric_cols:
        median_val = numeric_df[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        numeric_df[col] = numeric_df[col].fillna(median_val)

    n_clusters = int(min(8, max(2, np.sqrt(len(df) / 2))))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(numeric_df.values)
    label_series = pd.Series(labels, index=df.index)

    counts = label_series.value_counts().sort_index()
    cluster_ids = counts.index.to_numpy()
    cluster_probs = (counts / counts.sum()).values

    rng = np.random.default_rng(random_state)
    chosen_clusters = rng.choice(cluster_ids, size=n_rows, p=cluster_probs)

    sampled_indices = []
    for cluster_id in chosen_clusters:
        pool = label_series[label_series == cluster_id].index.to_numpy()
        sampled_indices.append(int(rng.choice(pool)))

    return df.loc[sampled_indices].reset_index(drop=True)


def generate_kde_sample(df: pd.DataFrame, n_rows: int, random_state: int = 42) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()

    rng = np.random.default_rng(random_state)
    synth = pd.DataFrame(index=range(n_rows))

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    for col in categorical_cols:
        probs = df[col].astype(str).value_counts(normalize=True)
        if len(probs) == 0:
            synth[col] = [np.nan] * n_rows
            continue
        synth[col] = rng.choice(probs.index.to_numpy(), size=n_rows, p=probs.values)

    if len(numeric_cols) == 0:
        return synth[df.columns]

    numeric_df = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    for col in numeric_cols:
        median_val = numeric_df[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        numeric_df[col] = numeric_df[col].fillna(median_val)

    X = numeric_df.values
    if len(X) < 3:
        return generate_bayesian_bootstrap_sample(df, n_rows, random_state=random_state)

    std_scale = float(np.mean(np.std(X, axis=0)))
    if not np.isfinite(std_scale) or std_scale <= 0:
        std_scale = 1.0
    bandwidth = max(0.05, std_scale * (len(X) ** (-1.0 / (X.shape[1] + 4))))

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(X)
    sampled_numeric = kde.sample(n_rows, random_state=random_state)

    for idx, col in enumerate(numeric_cols):
        col_vals = numeric_df[col].values
        lo, hi = float(np.nanmin(col_vals)), float(np.nanmax(col_vals))
        synth[col] = np.clip(sampled_numeric[:, idx], lo, hi)

    return synth[df.columns]


def compute_selected_comparison_metrics(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    selected_metrics: list[str],
    continuous_cols: list[str],
    categorical_cols: list[str],
):
    result = {}
    real_aligned, synth_aligned = _align_real_synth_for_metrics(real_df, synth_df)
    if real_aligned is None or synth_aligned is None:
        for metric_name in selected_metrics:
            result[metric_name] = np.nan
        return result

    needs_summary = any(m in selected_metrics for m in ['RMSE', 'MSE', 'MAE', 'AUC', 'Mean |SMD|'])
    summary = None
    if needs_summary:
        try:
            summary = EvaluationMetrics.summarize_metrics(real_aligned, synth_aligned)
        except Exception:
            summary = None

    for metric_name in selected_metrics:
        if metric_name == 'RMSE':
            result[metric_name] = summary['rmse'] if summary else np.nan
        elif metric_name == 'MSE':
            result[metric_name] = summary['mse'] if summary else np.nan
        elif metric_name == 'MAE':
            result[metric_name] = summary['mae'] if summary else np.nan
        elif metric_name == 'AUC':
            result[metric_name] = summary.get('auc', np.nan) if summary else np.nan
        elif metric_name == 'Mean |SMD|':
            if summary and 'smd' in summary and len(summary['smd']) > 0:
                result[metric_name] = float(np.mean(np.abs(list(summary['smd'].values()))))
            else:
                result[metric_name] = np.nan
        elif metric_name in ['Dimension-wise Distance', 'Manhattan (L1)']:
            cont_cols = [c for c in continuous_cols if c in real_aligned.columns and c in synth_aligned.columns]
            cat_cols = [c for c in categorical_cols if c in real_aligned.columns and c in synth_aligned.columns]
            distance_report = compute_dimension_wise_distance(real_aligned, synth_aligned, cont_cols, cat_cols)
            if distance_report is None:
                result[metric_name] = np.nan
            elif metric_name == 'Dimension-wise Distance':
                result[metric_name] = distance_report['dimension_wise_distance']
            else:
                result[metric_name] = distance_report['l1_sum']
        elif metric_name == 'k-Anonymity':
            try:
                k_report = EvaluationMetrics.compute_k_anonymity(synth_aligned)
                result[metric_name] = k_report['k_anonymity']
            except Exception:
                result[metric_name] = np.nan

    return result


def build_pdf_report_bytes(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    metrics: dict,
    config: dict,
    model,
    comparison_df: pd.DataFrame | None,
    selected_comparison_metrics: list[str],
):
    buffer = BytesIO()

    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.suptitle("SCOG Training Report", fontsize=18, fontweight='bold')
        plt.axis('off')

        summary_lines = [
            f"Timestamp: {pd.Timestamp.now()}",
            "",
            "Overall Metrics",
            f"- RMSE: {metrics.get('rmse', np.nan):.6f}",
            f"- MSE: {metrics.get('mse', np.nan):.6f}",
            f"- MAE: {metrics.get('mae', np.nan):.6f}",
            f"- AUC: {metrics.get('auc', np.nan)}",
            "",
            "Data Summary",
            f"- Real rows: {len(real_df)}",
            f"- Synthetic rows: {len(synth_df)}",
            f"- Columns: {len(real_df.columns)}",
            f"- Continuous: {len(config.get('continuous_cols', []))}",
            f"- Categorical: {len(config.get('categorical_cols', []))}",
            f"- Binary: {len(config.get('binary_cols', []))}",
            "",
            "Training Settings",
            f"- Epochs: {config.get('epochs', 'N/A')}",
            f"- Batch size: {config.get('batch_size', 'N/A')}",
            f"- Generator LR: {config.get('lr_g', 'N/A')}",
            f"- Discriminator LR: {config.get('lr_d', 'N/A')}",
            f"- Discriminator steps: {config.get('n_critic', 'N/A')}",
        ]
        fig.text(0.06, 0.9, "\n".join(summary_lines), va='top', fontsize=11)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        if model is not None and len(getattr(model, 'g_losses', [])) > 0 and len(getattr(model, 'd_losses', [])) > 0:
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.plot(model.g_losses, label='Generator Loss', color=CB_BLUE)
            ax.plot(model.d_losses, label='Discriminator Loss', color=CB_BROWN)
            ax.set_title('Training Loss Curves')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(alpha=0.2)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        numeric_cols = [c for c in real_df.select_dtypes(include=[np.number]).columns if c in synth_df.columns]
        if len(numeric_cols) > 0:
            max_plots = min(6, len(numeric_cols))
            cols_to_plot = numeric_cols[:max_plots]
            ncols = 2
            nrows = int(np.ceil(max_plots / ncols))
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11.69, 8.27))
            axes = np.array(axes).reshape(-1)
            for idx, col in enumerate(cols_to_plot):
                ax = axes[idx]
                real_vals = pd.to_numeric(real_df[col], errors='coerce').dropna().values
                synth_vals = pd.to_numeric(synth_df[col], errors='coerce').dropna().values
                ax.hist(real_vals, bins=30, alpha=0.5, label='Real', density=True, color=CB_BLUE)
                ax.hist(synth_vals, bins=30, alpha=0.5, label='Synthetic', density=True, color=CB_BROWN)
                ax.set_title(col)
                ax.legend(fontsize=8)
            for idx in range(max_plots, len(axes)):
                axes[idx].axis('off')
            fig.suptitle('Distribution Comparison (Numeric Columns)', fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        if len(numeric_cols) >= 2:
            real_corr = real_df[numeric_cols].apply(pd.to_numeric, errors='coerce').corr(method='spearman')
            synth_corr = synth_df[numeric_cols].apply(pd.to_numeric, errors='coerce').corr(method='spearman')
            diff_corr = (real_corr - synth_corr).abs()

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, corr, title in zip(
                axes,
                [real_corr, synth_corr, diff_corr],
                ['Real Correlation', 'Synthetic Correlation', 'Absolute Difference']
            ):
                im = ax.imshow(corr.values, vmin=-1 if title != 'Absolute Difference' else 0, vmax=1, cmap='BrBG', aspect='auto')
                ax.set_title(title)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle('Correlation Structure', fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        if comparison_df is not None and not comparison_df.empty:
            table_df = comparison_df.copy()
            for col in table_df.columns:
                if col != 'Method':
                    table_df[col] = pd.to_numeric(table_df[col], errors='coerce').round(4)

            metric_cols_all = [c for c in table_df.columns if c != 'Method']

            def _short_header(name: str) -> str:
                mapping = {
                    'Dimension-wise Distance': 'Dim-wise Dist',
                    'Manhattan (L1)': 'Manhattan L1',
                    'Mean |SMD|': 'Mean |SMD|',
                    'k-Anonymity': 'k-Anon',
                    'CTGAN (current model)': 'CTGAN',
                }
                return mapping.get(name, name)

            if len(metric_cols_all) > 0:
                table_cols = ['Method'] + metric_cols_all
                chunk_df = table_df[table_cols].copy()
                chunk_df['Method'] = chunk_df['Method'].astype(str).apply(lambda x: textwrap.fill(x, width=24))
                display_headers = [textwrap.fill(_short_header(col), width=14) for col in table_cols]

                fig = plt.figure(figsize=(11.69, 8.27))
                plt.axis('off')
                fig.suptitle('Method Comparison Summary', fontsize=16, fontweight='bold')

                method_col_width = 0.34
                metric_col_width = (0.96 - method_col_width) / len(metric_cols_all)
                col_widths = [method_col_width] + [metric_col_width] * len(metric_cols_all)

                table = plt.table(
                    cellText=chunk_df.values,
                    colLabels=display_headers,
                    cellLoc='center',
                    colLoc='center',
                    colWidths=col_widths,
                    bbox=[0.02, 0.06, 0.96, 0.82],
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 1.3)

                for (row, col), cell in table.get_celld().items():
                    if row == 0:
                        cell.set_facecolor('#e5e7eb')
                        cell.set_text_props(weight='bold', color='#111827')
                    elif col == 0:
                        cell.set_facecolor('#f8fafc')
                        cell.set_text_props(ha='left')
                    cell.set_edgecolor('#d1d5db')

                fig.text(
                    0.02,
                    0.02,
                    'Lower values are generally better for error/distance metrics. Interpret methods alongside metric definitions.',
                    fontsize=8,
                    color='#4b5563'
                )

                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

            metric_cols = [m for m in selected_comparison_metrics if m in comparison_df.columns]
            if len(metric_cols) > 0:
                nplots = len(metric_cols)
                ncols = 2
                nrows = int(np.ceil(nplots / ncols))
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11.69, 8.27))
                axes = np.array(axes).reshape(-1)

                for idx, metric_name in enumerate(metric_cols):
                    ax = axes[idx]
                    data = comparison_df[['Method', metric_name]].copy()
                    data[metric_name] = pd.to_numeric(data[metric_name], errors='coerce')
                    data = data.dropna(subset=[metric_name]).sort_values(metric_name)
                    ax.barh(data['Method'], data[metric_name], color=CB_BLUE)
                    ax.set_title(metric_name)
                    ax.tick_params(axis='y', labelsize=8)
                    ax.grid(axis='x', alpha=0.2)

                for idx in range(nplots, len(axes)):
                    axes[idx].axis('off')

                fig.suptitle('Comparison Metrics by Method', fontsize=14)
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()


def render_correlation_heatmap(corr_df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(corr_df.values, vmin=-1, vmax=1, cmap='BrBG', aspect='auto')
    ax.set_title(title)
    ax.set_xticks(np.arange(len(corr_df.columns)))
    ax.set_yticks(np.arange(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(corr_df.index, fontsize=8)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label('Spearman correlation')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_changed_hyperparameters(current_values: dict, default_values: dict, title: str = "Changes from defaults"):
    changed_rows = []
    for key, default_value in default_values.items():
        current_value = current_values.get(key)
        if current_value != default_value:
            changed_rows.append((key, current_value, default_value))

    st.subheader(title, anchor=False)
    if not changed_rows:
        st.caption("No hyperparameter changes from defaults.")
        return

    lines = []
    for name, current_value, default_value in changed_rows:
        lines.append(
            f"<li><span style='color:#dc2626; font-weight:600'>{name}: {current_value}</span> "
            f"<span style='color:#6b7280'>(default: {default_value})</span></li>"
        )
    st.markdown("<ul>" + "".join(lines) + "</ul>", unsafe_allow_html=True)


def render_lay_summary(text: str):
    st.markdown(f"<div class='math-lay-summary'>{text}</div>", unsafe_allow_html=True)


def render_symbol_definitions(symbol_rows: list[tuple[str, str, str]], title: str = "Symbol Definitions"):
    st.markdown(f"**{title}**")
    h1, h2, h3 = st.columns([1, 3, 4])
    h1.markdown("**Symbol**")
    h2.markdown("**Meaning**")
    h3.markdown("**Value Source**")

    for symbol_latex, meaning, source in symbol_rows:
        c1, c2, c3 = st.columns([1, 3, 4])
        c1.latex(symbol_latex)
        c2.write(meaning)
        c3.write(source)

# main title
st.markdown(
    '''<div class="main-header"><span style="color:#8C510A">SCOG</span>: <span><span style="color:#8C510A">S</span>ynthetic <span style="color:#8C510A">Co</span>ntrol <span style="color:#8C510A">G</span>enerator using CTGAN</span></div>''',
    unsafe_allow_html=True
)

# sidebar for navigation
blink_candidates = [
    Path(__file__).parent / "github_site" / "SCOG_pics" / "head_blink_gif.gif",
    Path(__file__).parent / "SCOG_pics" / "head_blink_gif.gif"
]
blink_data_uri = None
for blink_path in blink_candidates:
    if blink_path.exists():
        with open(blink_path, "rb") as blink_file:
            blink_data_uri = base64.b64encode(blink_file.read()).decode("utf-8")
        break

logo_candidates = [
    Path(__file__).parent / "github_site" / "SCOG_pics" / "logo.png",
    Path(__file__).parent / "SCOG_pics" / "logo.png"
]
logo_data_uri = None
for logo_path in logo_candidates:
    if logo_path.exists():
        with open(logo_path, "rb") as logo_file:
            logo_data_uri = base64.b64encode(logo_file.read()).decode("utf-8")
        break

sidebar_visual_html = ""
if blink_data_uri:
    sidebar_visual_html += f"<div class='sidebar-blink-wrap'><img src='data:image/gif;base64,{blink_data_uri}' alt='SCOG animated head'></div>"
if logo_data_uri:
    sidebar_visual_html += f"<div class='sidebar-logo-wrap'><img src='data:image/png;base64,{logo_data_uri}' alt='SCOG logo'></div>"

if sidebar_visual_html:
    st.sidebar.markdown(sidebar_visual_html, unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Upload Data", "Preprocessing", "Configure Model", "Train Model", "View Results", "Method Comparison", "Download", "Resources"]
)

# page 1: upload data

if page == "Upload Data":
    st.markdown('<div class="section-header">Upload Your Data</div>', unsafe_allow_html=True)

    app_dir = Path(__file__).parent
    aim2_dir = app_dir.parent.parent / "Aim 2"
    saved_rdata_dir = app_dir.parent.parent / "Saved_Rdata"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload CSV File", anchor=False)
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv']
        )
        
        if uploaded_file is not None:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.original_df = st.session_state.df.copy()
            st.success(f"File uploaded: {uploaded_file.name}")
    
    with col2:
        st.subheader("Load Sample Data", anchor=False)
        sample_files = [
            ("Sample Data 1", "testdata4.csv"),
            ("Sample Data 2", "testdata5.csv"),
            ("Sample Data 3", "testdata6.csv"),
        ]

        for button_label, filename in sample_files:
            if st.button(button_label):
                candidates = [app_dir / filename, aim2_dir / filename]
                resolved_path = next((path for path in candidates if path.exists()), None)

                if resolved_path is not None:
                    st.session_state.df = pd.read_csv(resolved_path)
                    st.session_state.original_df = st.session_state.df.copy()
                    st.success(f"Loaded {resolved_path.name}")
                else:
                    st.error(f"{filename} not found")

    st.markdown("---")
    st.subheader("Load Real Data", anchor=False)
    st.caption("Real data is simulated based on the data characteristics from Crohn's Disease and COVID-19 clinical sources. More information on the data sources can be found in the Resources tab.")

    def load_real_dataset(filename: str, success_label: str):
        resolved_path = saved_rdata_dir / filename

        if resolved_path.exists():
            st.session_state.df = pd.read_csv(resolved_path)
            st.session_state.original_df = st.session_state.df.copy()
            st.success(f"Loaded {success_label}")
            return True

        st.error(f"{filename} not found in Saved_Rdata")
        return False

    def load_combined_datasets(selected_labels: list[str], file_map: dict[str, str], disease_name: str):
        if len(selected_labels) < 2:
            st.warning(f"Select at least two {disease_name} datasets to combine.")
            return

        loaded_frames = []
        missing_files = []

        for label in selected_labels:
            filename = file_map[label]
            resolved_path = saved_rdata_dir / filename
            if resolved_path.exists():
                loaded_frames.append(pd.read_csv(resolved_path))
            else:
                missing_files.append(filename)

        if missing_files:
            st.error(f"Missing files: {', '.join(missing_files)}")
            return

        common_cols = list(loaded_frames[0].columns)
        for frame in loaded_frames[1:]:
            common_cols = [col for col in common_cols if col in frame.columns]

        if not common_cols:
            st.error("Selected datasets do not share any common columns.")
            return

        combined_df = pd.concat([frame[common_cols] for frame in loaded_frames], ignore_index=True)
        st.session_state.df = combined_df
        st.session_state.original_df = combined_df.copy()

        total_rows = sum(len(frame) for frame in loaded_frames)
        st.success(
            f"Loaded combined {disease_name} data from {len(selected_labels)} datasets "
            f"({total_rows} rows, {len(common_cols)} common columns)."
        )

        dropped_cols = sorted({col for frame in loaded_frames for col in frame.columns if col not in common_cols})
        if dropped_cols:
            st.info(
                "Only shared columns were kept when combining. "
                f"Dropped non-shared columns: {', '.join(dropped_cols[:10])}"
                + (" ..." if len(dropped_cols) > 10 else "")
            )

    real_data_files = [
        ("Real Data, External", "crohns_external.csv"),
        ("Real Data, Observational", "crohns_observational.csv"),
        ("Real Data, RCT", "crohns_rct.csv"),
    ]
    covid_data_files = [
        ("COVID-19, RCT", "covid_rct.csv"),
        ("COVID-19, Observational", "covid_ob.csv"),
        ("COVID-19, External", "covid_ex.csv"),
    ]

    real_col1, real_col2 = st.columns(2)

    with real_col1:
        st.markdown("**Crohn's Disease**")
        for button_label, filename in real_data_files:
            if st.button(button_label, key=f"real_data_{filename}"):
                load_real_dataset(filename, button_label)

        crohns_options = [label for label, _ in real_data_files]
        crohns_file_map = {label: filename for label, filename in real_data_files}
        selected_crohns = st.multiselect(
            "Select Crohn's datasets to combine",
            options=crohns_options,
            key="combine_crohns_select",
        )
        if st.button("Load Combined Crohn's Data", key="combine_crohns_button"):
            load_combined_datasets(selected_crohns, crohns_file_map, "Crohn's Disease")

    with real_col2:
        st.markdown("**COVID-19**")
        for button_label, filename in covid_data_files:
            if st.button(button_label, key=f"real_data_{filename}"):
                load_real_dataset(filename, button_label)

        covid_options = [label for label, _ in covid_data_files]
        covid_file_map = {label: filename for label, filename in covid_data_files}
        selected_covid = st.multiselect(
            "Select COVID-19 datasets to combine",
            options=covid_options,
            key="combine_covid_select",
        )
        if st.button("Load Combined COVID-19 Data", key="combine_covid_button"):
            load_combined_datasets(selected_covid, covid_file_map, "COVID-19")
    
    # display data preview if loaded
    if st.session_state.df is not None:
        st.subheader("Data Preview", anchor=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Dimensions**: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns")
        with col2:
            st.write(f"**Memory Usage**: {st.session_state.df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        # display first few rows
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        
        # show statistics
        with st.expander("Data Statistics"):
            st.write(st.session_state.df.describe())
        
# page 2: preprocessing

elif page == "Preprocessing":
    st.markdown('<div class="section-header">Preprocessing</div>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.error("Please upload data first.")
    else:
        # missing value assessment
        st.subheader("Missing Values Assessment", anchor=False)
        missing = st.session_state.df.isnull().sum()
        if missing.sum() > 0:
            st.markdown('<div class="warning-box"><b>Missing Values Detected:</b><br>', unsafe_allow_html=True)
            for col, count in missing[missing > 0].items():
                st.markdown(f"- {col}: {count} missing values")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("**Optional Missing Data Handling**")
            st.caption("Choose one action below, then click Apply. This updates the active dataset used for training.")

            st.session_state.missing_data_action = st.radio(
                "Missing data action",
                options=['drop', 'impute'],
                index=0 if st.session_state.missing_data_action == 'drop' else 1,
                format_func=lambda x: "Drop rows with missing values" if x == 'drop' else "Impute missing values",
                horizontal=True,
            )

            if st.session_state.missing_data_action == 'impute':
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.session_state.missing_num_impute = st.selectbox(
                        "Numeric imputation",
                        options=['median', 'mean', 'zero', 'min', 'max', 'random_sample'],
                        index=['median', 'mean', 'zero', 'min', 'max', 'random_sample'].index(
                            st.session_state.missing_num_impute if st.session_state.missing_num_impute in ['median', 'mean', 'zero', 'min', 'max', 'random_sample'] else 'median'
                        ),
                        help="How to impute missing values in numeric columns."
                    )
                with c2:
                    st.session_state.missing_cat_impute = st.selectbox(
                        "Categorical imputation",
                        options=['mode', 'constant', 'random_sample'],
                        index=['mode', 'constant', 'random_sample'].index(
                            st.session_state.missing_cat_impute if st.session_state.missing_cat_impute in ['mode', 'constant', 'random_sample'] else 'mode'
                        ),
                        help="How to impute missing values in categorical columns."
                    )
                with c3:
                    st.session_state.missing_cat_fill_value = st.text_input(
                        "Categorical fill value",
                        value=st.session_state.missing_cat_fill_value,
                        help="Used for constant categorical imputation, or as fallback when mode/random cannot be computed."
                    )

            if st.button("Apply Missing Data Action"):
                current_df_before_action = st.session_state.df.copy()
                before_rows = len(st.session_state.df)
                before_missing_cells = int(st.session_state.df.isnull().sum().sum())
                before_missing_cols = int((st.session_state.df.isnull().sum() > 0).sum())
                work_df = st.session_state.df.copy()

                if st.session_state.missing_data_action == 'drop':
                    work_df = work_df.dropna().reset_index(drop=True)
                else:
                    for col in work_df.columns:
                        if work_df[col].isnull().sum() == 0:
                            continue
                        if pd.api.types.is_numeric_dtype(work_df[col]):
                            if st.session_state.missing_num_impute == 'mean':
                                fill_value = work_df[col].mean()
                            elif st.session_state.missing_num_impute == 'zero':
                                fill_value = 0.0
                            elif st.session_state.missing_num_impute == 'min':
                                fill_value = work_df[col].min()
                            elif st.session_state.missing_num_impute == 'max':
                                fill_value = work_df[col].max()
                            elif st.session_state.missing_num_impute == 'random_sample':
                                observed = work_df[col].dropna()
                                if len(observed) > 0:
                                    sampled = observed.sample(n=work_df[col].isnull().sum(), replace=True, random_state=42).values
                                    work_df.loc[work_df[col].isnull(), col] = sampled
                                    continue
                                fill_value = 0.0
                            else:
                                fill_value = work_df[col].median()
                            if pd.isna(fill_value):
                                fill_value = 0.0
                        else:
                            if st.session_state.missing_cat_impute == 'constant':
                                fill_value = st.session_state.missing_cat_fill_value
                            elif st.session_state.missing_cat_impute == 'random_sample':
                                observed = work_df[col].dropna()
                                if len(observed) > 0:
                                    sampled = observed.sample(n=work_df[col].isnull().sum(), replace=True, random_state=42).values
                                    work_df.loc[work_df[col].isnull(), col] = sampled
                                    continue
                                fill_value = st.session_state.missing_cat_fill_value
                            else:
                                modes = work_df[col].mode(dropna=True)
                                fill_value = modes.iloc[0] if len(modes) > 0 else st.session_state.missing_cat_fill_value
                        work_df[col] = work_df[col].fillna(fill_value)

                st.session_state.df = work_df
                after_rows = len(st.session_state.df)
                after_missing_cells = int(st.session_state.df.isnull().sum().sum())
                after_missing_cols = int((st.session_state.df.isnull().sum() > 0).sum())

                if after_rows < 2:
                    st.session_state.df = current_df_before_action
                    st.error(
                        "This action would leave fewer than 2 rows, so no changes were applied. "
                        "Please use less aggressive preprocessing settings."
                    )
                    st.stop()

                summary_df = pd.DataFrame([
                    {
                        'stage': 'before',
                        'rows': before_rows,
                        'missing_cells': before_missing_cells,
                        'columns_with_missing': before_missing_cols,
                    },
                    {
                        'stage': 'after',
                        'rows': after_rows,
                        'missing_cells': after_missing_cells,
                        'columns_with_missing': after_missing_cols,
                    }
                ])
                st.dataframe(summary_df, width='stretch', hide_index=True)

                if st.session_state.missing_data_action == 'drop':
                    st.success(f"Dropped rows with missing values. Rows: {before_rows} → {after_rows}")
                else:
                    remaining_na = int(st.session_state.df.isnull().sum().sum())
                    st.success(f"Imputation complete. Remaining missing cells: {remaining_na}")
        else:
            st.markdown('<div class="success-box"> No missing values</div>', unsafe_allow_html=True)

        # optional outlier detection/removal
        st.subheader("Outlier Detection and Removal", anchor=False)
        st.session_state.outlier_enabled = st.checkbox(
            "Enable outlier detection/removal",
            value=st.session_state.outlier_enabled
        )

        if st.session_state.outlier_enabled:
            numeric_candidates = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_candidates:
                st.info("No numeric columns available for outlier detection.")
            else:
                st.session_state.outlier_combination_rule = st.radio(
                    "Outlier row removal rule",
                    options=['OR', 'AND'],
                    index=0 if st.session_state.outlier_combination_rule == 'OR' else 1,
                    horizontal=True,
                    help=(
                        "OR: remove a row if it is an outlier in ANY selected column. "
                        "AND: remove only if it is an outlier in ALL selected columns."
                    )
                )
                if st.session_state.outlier_combination_rule == 'OR':
                    st.caption("OR rule: a row is removed when any selected column flags it as an outlier.")
                else:
                    st.caption("AND rule: a row is removed only when all selected columns flag it as an outlier.")

                st.session_state.outlier_iqr_multiplier = st.slider(
                    "IQR multiplier",
                    min_value=1.0,
                    max_value=3.0,
                    value=float(st.session_state.outlier_iqr_multiplier),
                    step=0.1,
                    help="Rows outside [Q1 - k*IQR, Q3 + k*IQR] are flagged as outliers."
                )

                st.session_state.outlier_columns = st.multiselect(
                    "Columns to check for outliers",
                    options=numeric_candidates,
                    default=[c for c in st.session_state.outlier_columns if c in numeric_candidates],
                    help="Column-by-column decision: choose exactly which columns participate in outlier removal."
                )

                if st.session_state.outlier_columns:
                    summary_rows = []
                    masks = []
                    for col in st.session_state.outlier_columns:
                        col_mask = compute_iqr_outlier_mask(st.session_state.df[col], st.session_state.outlier_iqr_multiplier)
                        masks.append(col_mask)
                        out_count = int(col_mask.sum())
                        out_pct = float((out_count / max(1, len(st.session_state.df))) * 100)
                        summary_rows.append({
                            'column': col,
                            'outlier_rows': out_count,
                            'outlier_pct': round(out_pct, 2),
                        })

                    if st.session_state.outlier_combination_rule == 'OR':
                        combined_mask = masks[0].copy()
                        for mask in masks[1:]:
                            combined_mask = combined_mask | mask
                    else:
                        combined_mask = masks[0].copy()
                        for mask in masks[1:]:
                            combined_mask = combined_mask & mask

                    st.dataframe(pd.DataFrame(summary_rows), width='stretch')
                    st.caption(
                        f"Rows flagged in any selected column: {int(combined_mask.sum())} / {len(st.session_state.df)}"
                    )

                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("Apply Outlier Removal", help="Remove rows flagged as outliers in selected columns"):
                            kept_df = st.session_state.df.loc[~combined_mask].reset_index(drop=True)
                            removed = len(st.session_state.df) - len(kept_df)
                            if len(kept_df) < 2:
                                st.error(
                                    "Outlier removal would leave fewer than 2 rows, so no changes were applied. "
                                    "Relax the rule/settings or choose fewer columns."
                                )
                            else:
                                st.session_state.df = kept_df
                                st.success(f"Removed {removed} outlier rows. New row count: {len(st.session_state.df)}")
                    with col_b:
                        if st.button("Restore Original Data", help="Revert to the dataset as initially uploaded/loaded"):
                            if st.session_state.original_df is not None:
                                st.session_state.df = st.session_state.original_df.copy()
                                st.success(f"Restored original dataset ({len(st.session_state.df)} rows).")
                            else:
                                st.info("No original dataset snapshot available to restore.")

        # privacy assessment
        st.subheader("Privacy Assessment", anchor=False)
        with st.expander("k-Anonymity", expanded=False):
            st.write("**k-Anonymity** measures the minimum number of rows sharing same quasi-identifier values." )
            st.write("This is useful for ensuring individual patients cannot be identified from the dataset.")
            st.write("Higher k = better privacy. k >= 3 is considered reasonably safe.")

            k_anon_result = EvaluationMetrics.compute_k_anonymity(st.session_state.df)

            col1, col2, col3 = st.columns(3)
            with col1:
                k_val = k_anon_result['k_anonymity']
                if k_val == float('inf'):
                    st.metric("k-Anonymity", "∞", help="Perfect privacy (no duplicate combinations)")
                else:
                    st.metric("k-Anonymity", f"{k_val}", help="Minimum group size for quasi-identifiers")
            with col2:
                vulnerable = k_anon_result['vulnerable_count']
                st.metric("Vulnerable Rows", f"{vulnerable}", help="Rows in groups with k < 3")
            with col3:
                coverage = k_anon_result['coverage']
                st.metric("Safe Coverage", f"{coverage:.1f}%", help="% of rows in groups with k >= 3")

            st.info(f"Quasi-identifiers used: {', '.join(k_anon_result['quasi_identifiers_used'])}")

        # optional correlation testing model selection
        st.subheader("Correlation Testing Model", anchor=False)
        st.session_state.use_correlation_model = st.checkbox(
            "Enable correlation-aware synthetic model",
            value=st.session_state.use_correlation_model,
            help="Uses a separate training model that adds correlation-preservation loss. Default model remains unchanged when disabled."
        )

        correlation_report = compute_correlation_report(st.session_state.df)
        st.session_state.correlation_baseline = correlation_report
        if correlation_report is None:
            st.info("Correlation testing needs at least 2 numeric columns.")
        else:
            st.caption(
                f"Numeric columns for Spearman correlation: {len(correlation_report['numeric_cols'])} | "
                f"Mean absolute baseline correlation: {correlation_report['real_mean_abs_corr']:.3f}"
            )
            with st.expander("View baseline correlation matrix (real data)", expanded=False):
                st.dataframe(correlation_report['real_corr'], width='stretch')

        # column classification
        st.subheader("Column Classification", anchor=False)
        st.write("This auto-classifies the data type of each column. You can manually adjust the data type for any columns that were auto-classified incorrectly.")
        if st.button("Classify Columns", help="Automatically detect column types"):
            continuous = []
            categorical = []
            binary = []

            for col in st.session_state.df.columns:
                nunique = st.session_state.df[col].nunique()
                if nunique == 2:
                    binary.append(col)
                elif 3 <= nunique <= 10:
                    categorical.append(col)
                else:
                    continuous.append(col)

            st.session_state.continuous_cols = continuous
            st.session_state.categorical_cols = categorical
            st.session_state.binary_cols = binary
            st.success("Columns classified.")

        col1, col2, col3 = st.columns(3)
        with col1:
            continuous_cols = st.multiselect(
                "Continuous Columns",
                options=st.session_state.df.columns.tolist(),
                default=st.session_state.continuous_cols,
                help="Columns with continuous numeric values"
            )
            st.session_state.continuous_cols = continuous_cols
        with col2:
            categorical_cols = st.multiselect(
                "Categorical Columns",
                options=st.session_state.df.columns.tolist(),
                default=st.session_state.categorical_cols,
                help="Columns with 3-10 categories"
            )
            st.session_state.categorical_cols = categorical_cols
        with col3:
            binary_cols = st.multiselect(
                "Binary Columns",
                options=st.session_state.df.columns.tolist(),
                default=st.session_state.binary_cols,
                help="Columns with exactly 2 values"
            )
            st.session_state.binary_cols = binary_cols

        # condition column selection
        st.subheader("Condition Column (Optional)", anchor=False)
        st.write("Select a column for conditional generation.")
        st.write("This allows generating synthetic data based on specific values for that column.")
        st.write("In randomised controlled trials, this column typically contains treatment assignments.")
        with st.expander("When to use a conditional column", expanded=False):
            st.write("""
            Use a conditional column when you have a treatment/intervention assignment column (e.g., 'Treatment', 'Group').
            Binary columns are ideal, but small categorical columns (multi-arm trials) can also work.
            """)

        condition_col = st.selectbox(
            "Choose condition column",
            options=[None] + st.session_state.df.columns.tolist(),
            index=([None] + st.session_state.df.columns.tolist()).index(st.session_state.condition_col) if st.session_state.condition_col in st.session_state.df.columns.tolist() else 0,
            format_func=lambda x: "No condition column" if x is None else x,
            help="Leave as 'No condition column' if you don't need conditional generation"
        )
        st.session_state.condition_col = condition_col

        if condition_col is not None:
            nunique = st.session_state.df[condition_col].nunique()
            if nunique == 2:
                st.info(f"Binary column - valid ({nunique} unique values)")
            elif 3 <= nunique <= 10:
                st.info(f"Categorical column - valid ({nunique} unique values)")
            else:
                st.warning(f"Warning: Too many unique values ({nunique}), may not work well as condition")


# page 3: configure model

elif page == "Configure Model":
    st.markdown('<div class="section-header">Configure Model Parameters</div>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.error("Please upload data first.")
    else:
        condition_col = st.session_state.get('condition_col', None)
        
        # model hyperparameters
        st.subheader("Categorical Encoding Option", anchor=False)
        st.write("Choose how categorical variables are represented for model training.")
        encoding_mode = st.radio(
            "Encoding mode",
            options=["onehot", "ordinal"],
            format_func=lambda x: "One-hot (default model)" if x == "onehot" else "Ordinal model (alternative)",
            horizontal=True,
            help="Default keeps one-hot encoding. Ordinal mode treats selected columns as ordered categories."
        )

        ordinal_orders = {}
        if encoding_mode == "ordinal":
            ordinal_candidates = [c for c in (st.session_state.categorical_cols + st.session_state.binary_cols) if c != condition_col]
            st.session_state.ordinal_cols = st.multiselect(
                "Ordinal Columns",
                options=ordinal_candidates,
                default=[c for c in st.session_state.ordinal_cols if c in ordinal_candidates],
                help="Select only truly ordered categorical columns (e.g., mild<moderate<severe)."
            )

            if st.session_state.ordinal_cols:
                st.markdown("**Ordinal Category Order (low → high)**")
                st.caption("Adjust order using comma-separated labels. Must include each detected category exactly once.")
                for ord_col in st.session_state.ordinal_cols:
                    detected = list(pd.Series(st.session_state.df[ord_col]).dropna().unique())
                    default_order = st.session_state.ordinal_orders.get(ord_col, detected)
                    order_text = st.text_input(
                        f"{ord_col} order",
                        value=", ".join([str(v) for v in default_order]),
                        key=f"ordinal_order_{ord_col}",
                        help=f"Detected categories: {detected}"
                    )
                    parsed = [v.strip() for v in order_text.split(',') if v.strip() != ""]
                    detected_set = {str(v) for v in detected}
                    parsed_set = {str(v) for v in parsed}
                    if parsed and parsed_set == detected_set and len(parsed) == len(detected):
                        normalized = []
                        for value in parsed:
                            matched = next((orig for orig in detected if str(orig) == value), value)
                            normalized.append(matched)
                        ordinal_orders[ord_col] = normalized
                    else:
                        st.warning(f"Using detected order for '{ord_col}' because the custom order is invalid.")
                        ordinal_orders[ord_col] = detected
                    st.write(f"Active order for {ord_col}: {ordinal_orders[ord_col]}")
            st.session_state.ordinal_orders = ordinal_orders
        else:
            st.session_state.ordinal_cols = []
            st.session_state.ordinal_orders = {}

        # model hyperparameters
        st.subheader("Model Hyperparameters", anchor=False)
        #st.caption("Column classification and condition column are configured in the Preprocessing tab.")


        with st.expander("Hyperparameter Tuning Guide", expanded=False):
            st.write("""
            **Noise Dimension:** Controls generator randomness. 
            Larger = more diversity, but may be slower.
            - For small datasets (n < 1000): use 32-64
            - For large datasets (n > 10000): use 64-128
            
            **Batch Size:** Number of samples per training step. 
            Larger = faster, but uses more memory.
            - Recommended: 10-25% of dataset size (set automatically)
            
            
            **Discriminator Steps:** How many times discriminator trains per generator step.
            Default is 5, which is a common choice for stable training.
            - Increase to 6-10 if synthetic quality is poor
            - Use 1-2 for faster, but less robust training
            
            **Learning Rates:** How fast models update the weight of the generator and discriminator.
            Having the same rate (G=D) is most stable.
            - Increase to 5e-4 if training is slow
            - Decrease to 1e-4 if training is unstable (large loss swings)
            
            **PAC (Packing):** Prevents mode collapse.
            Mode collapse happens when the generator ignores diversity. 
            1 is the standard.
            A higher PAC will result in slower training.
            - Increase to 2-4 if mode collapse occurs (repeated patterns in synthetic data)
            
            """)
        
        col1, col2 = st.columns(2)
        with col1:
            noise_dim = st.slider(
                "Noise Dimension",
                min_value=32,
                max_value=256,
                value=64,
                step=32,
                help="Larger = more diverse synthetic samples. Range: 32-256. Default: 64."
            )
            
            batch_size = st.slider(
                "Batch Size",
                min_value=16,
                max_value=512,
                value=max(32, len(st.session_state.df) // 16),
                step=16,
                help="Samples per training step. ~10-25% of data size is typical. Larger = faster, but uses more memory."
            )
            
            n_critic = st.slider(
                "Discriminator Steps",
                min_value=1,
                max_value=10,
                value=5,
                help="Discriminator training iterations per generator step. Increase to 6-10 if quality is poor."
            )
        
        with col2:
            lr_g = st.select_slider(
                "Generator Learning Rate",
                options=[1e-4, 2e-4, 5e-4, 1e-3, 2e-3],
                value=2e-4,
                help="How fast generator learns. Increase if training stalls, decrease if unstable."
            )
            
            lr_d = st.select_slider(
                "Discriminator Learning Rate",
                options=[1e-4, 2e-4, 5e-4, 1e-3, 2e-3],
                value=2e-4,
                help="How fast discriminator learns. Keep equal to generator LR for stability. Increase both if stuck."
            )
            
            pac = st.slider(
                "PAC (Packing)",
                min_value=1,
                max_value=10,
                value=1,
                help="Mode collapse prevention. 1=default. Increase to 2-4 if synthetic data has repeated patterns."
            )
        
        # model architecture
        st.subheader("Model Architecture", anchor=False)
        with st.expander("Architecture Sizing Guide", expanded=False):
            st.write("""
            **When to adjust architecture:**
            
            **Small datasets (< 1000 rows):**
            - Use smaller layers: 64-128
            - Reduces overfitting and training time
            
            **Medium datasets (1000-10000 rows):**
            - Default 256 is good balance
            - Equal sizes for generator and discriminator recommended
            
            **Large datasets (> 10000 rows):**
            - Can use 256-512
            - Larger discriminator (384-512) helps with complex distributions
            
            **General rule:**
            - Equal sizes is simplest approach
            - Larger layers = more complex, but slower training
            - Keep generator >= discriminator for stability
            
            """)
        
        col1, col2 = st.columns(2)
        with col1:
            gen_dim1 = st.slider(
                "Generator Hidden Layer 1",
                min_value=64,
                max_value=512,
                value=256,
                step=64,
                help="First hidden layer size. Default 256. Larger = more complex but slower."
            )
            gen_dim2 = st.slider(
                "Generator Hidden Layer 2",
                min_value=64,
                max_value=512,
                value=256,
                step=64,
                help="Second hidden layer size. Keep >= first layer for data flow."
            )
        
        with col2:
            disc_dim1 = st.slider(
                "Discriminator Hidden Layer 1",
                min_value=64,
                max_value=512,
                value=256,
                step=64,
                help="Discriminator layer 1. Can be <= generator for cost savings."
            )
            disc_dim2 = st.slider(
                "Discriminator Hidden Layer 2",
                min_value=64,
                max_value=512,
                value=256,
                step=64,
                help="Discriminator layer 2. Keep equal or smaller than layer 1."
            )
        
        # training parameters
        st.subheader("Training Parameters", anchor=False)
        with st.expander("Training Parameters Guide", expanded=False):
            st.write("""
            **Max Epochs:** Upper limit on training iterations.
            - Small data (< 1000): 50-100 epochs
            - Medium data: 100-200 epochs
            - Large data (> 10000): 200-300 epochs 
            - Note: early stopping halts training when no imrpovement occurs. The minimum epochs before early stopping occurs is 50.
            
            **Early Stopping Patience:** How many epochs with no improvement before stopping.
            - 3 stops the model early, making training faster 
            - 5-10 is a good balance for most data
            - 20 is larger and takes longer, but may help with more complex data
            
            **Min Delta (Improvement):** Smallest loss decrease to count as improvement.
            - 1e-4 (default): stops if plateau is real
            - 1e-3: allows more training even with small improvements
            - 1e-2: trains for much longer, which may help with more complex data
            
        
            """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            epochs = st.slider(
                "Max Epochs",
                min_value=50,
                max_value=500,
                value=100,
                step=10,
                help="Training runs >=50 epochs, stopping early if no improvement. Set to 100-300 depending on data size."
            )
        
        with col2:
            patience = st.slider(
                "Early Stopping Patience",
                min_value=1,
                max_value=20,
                value=3,
                help="Epochs with no improvement before stopping (min 50 epochs always run)."
            )
        
        with col3:
            min_delta = st.select_slider(
                "Min Delta (Improvement)",
                options=[1e-5, 1e-4, 1e-3, 1e-2],
                value=1e-4,
                help="Minimum loss decrease to count as improvement. "
            )
        
        # summary
        st.subheader("Configuration Summary", anchor=False)
        summary_text = textwrap.dedent(f"""
        **Data**: {len(st.session_state.df)} rows, {len(st.session_state.df.columns)} columns

        **Column Types:**
        - Continuous: {len(st.session_state.continuous_cols)} columns
        - Categorical: {len(st.session_state.categorical_cols)} columns
        - Binary: {len(st.session_state.binary_cols)} columns
        - Ordinal: {len(st.session_state.ordinal_cols)} columns
        - Condition Column: {condition_col if condition_col else "None"}
        - Encoding Mode: {encoding_mode}

        **Architecture:**
        - Generator: {noise_dim} → {gen_dim1} → {gen_dim2} → output
        - Discriminator: input → {disc_dim1} → {disc_dim2} → 1

        **Training:**
        - Max Epochs: {epochs} | Patience: {patience} | Min Delta: {min_delta}
        - Learning Rates: G={lr_g}, D={lr_d}
        - Batch Size: {batch_size}
        """)
        st.info(summary_text)

        configure_defaults = {
            'noise_dim': 64,
            'batch_size': max(32, len(st.session_state.df) // 16),
            'n_critic': 5,
            'lr_g': 2e-4,
            'lr_d': 2e-4,
            'pac': 1,
            'generator_dim': (256, 256),
            'discriminator_dim': (256, 256),
            'epochs': 100,
            'patience': 3,
            'min_delta': 1e-4,
            'use_correlation_model': st.session_state.use_correlation_model,
            'correlation_loss_weight': 1.0,
            'encoding_mode': 'onehot',
            'ordinal_cols': [],
        }

        correlation_loss_weight = 1.0
        if st.session_state.use_correlation_model:
            correlation_loss_weight = st.slider(
                "Correlation Preservation Weight (optional model)",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Higher values prioritize matching real-data correlation structure more strongly."
            )

        configure_current = {
            'noise_dim': noise_dim,
            'batch_size': batch_size,
            'n_critic': n_critic,
            'lr_g': lr_g,
            'lr_d': lr_d,
            'pac': pac,
            'generator_dim': (gen_dim1, gen_dim2),
            'discriminator_dim': (disc_dim1, disc_dim2),
            'epochs': epochs,
            'patience': patience,
            'min_delta': min_delta,
            'use_correlation_model': st.session_state.use_correlation_model,
            'correlation_loss_weight': correlation_loss_weight,
            'encoding_mode': encoding_mode,
            'ordinal_cols': list(st.session_state.ordinal_cols),
        }
        render_changed_hyperparameters(configure_current, configure_defaults)
        
        # save configuration to session
        st.session_state.config = {
            'continuous_cols': st.session_state.continuous_cols,
            'categorical_cols': st.session_state.categorical_cols,
            'binary_cols': st.session_state.binary_cols,
            'condition_col': condition_col,
            'noise_dim': noise_dim,
            'generator_dim': (gen_dim1, gen_dim2),
            'discriminator_dim': (disc_dim1, disc_dim2),
            'batch_size': batch_size,
            'n_critic': n_critic,
            'lr_g': lr_g,
            'lr_d': lr_d,
            'pac': pac,
            'epochs': epochs,
            'patience': patience,
            'min_delta': min_delta,
            'use_correlation_model': st.session_state.use_correlation_model,
            'correlation_loss_weight': correlation_loss_weight,
            'encoding_mode': encoding_mode,
            'ordinal_cols': list(st.session_state.ordinal_cols),
            'ordinal_orders': dict(st.session_state.ordinal_orders),
        }
        
        st.success("Configuration ready. Go to 'Train Model' page to start training.")


# page 4: train model

elif page == "Train Model":
    st.markdown('<div class="section-header">Train Model</div>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.error("Please upload data first.")
    elif not hasattr(st.session_state, 'config'):
        st.error(".Please configure model first.")
    else:
        st.subheader("Model Status: Ready to Train", anchor=False)
        
        # display configuration
        with st.expander("Current Configuration", expanded=False):
            st.json({
                'rows': len(st.session_state.df),
                'columns': len(st.session_state.df.columns),
                'continuous': len(st.session_state.config['continuous_cols']),
                'categorical': len(st.session_state.config['categorical_cols']),
                'binary': len(st.session_state.config['binary_cols']),
                'max_epochs': st.session_state.config['epochs'],
                'patience': st.session_state.config['patience']
            })

        with st.expander("Mathematical View of Current GAN Setup", expanded=False):
            cfg = st.session_state.config
            g_dim1, g_dim2 = cfg['generator_dim']
            d_dim1, d_dim2 = cfg['discriminator_dim']
            cond_col = cfg.get('condition_col')
            data_dim_train = len(st.session_state.df.columns)
            gp_lambda_train = 10.0

            train_defaults = {
                'noise_dim': 64,
                'batch_size': max(32, len(st.session_state.df) // 16),
                'n_critic': 5,
                'lr_g': 2e-4,
                'lr_d': 2e-4,
                'pac': 1,
                'generator_dim': (256, 256),
                'discriminator_dim': (256, 256),
                'epochs': 100,
                'patience': 3,
                'min_delta': 1e-4,
                'use_correlation_model': False,
                'correlation_loss_weight': 1.0,
                'encoding_mode': 'onehot',
                'ordinal_cols': [],
            }
            train_current = {
                'noise_dim': cfg['noise_dim'],
                'batch_size': cfg['batch_size'],
                'n_critic': cfg['n_critic'],
                'lr_g': cfg['lr_g'],
                'lr_d': cfg['lr_d'],
                'pac': cfg['pac'],
                'generator_dim': cfg['generator_dim'],
                'discriminator_dim': cfg['discriminator_dim'],
                'epochs': cfg['epochs'],
                'patience': cfg['patience'],
                'min_delta': cfg['min_delta'],
                'use_correlation_model': cfg.get('use_correlation_model', False),
                'correlation_loss_weight': cfg.get('correlation_loss_weight', 1.0),
                'encoding_mode': cfg.get('encoding_mode', 'onehot'),
                'ordinal_cols': cfg.get('ordinal_cols', []),
            }
            render_changed_hyperparameters(train_current, train_defaults)

            if cfg.get('use_correlation_model', False):
                st.info(
                    f"Optional correlation-aware model is ENABLED (weight={cfg.get('correlation_loss_weight', 1.0):.2f}). "
                    "Default CTGAN is bypassed for this run."
                )
            else:
                st.info("Using default CTGAN model (correlation-aware mode disabled).")

            show_math_explanations = st.checkbox(
                "Show equation explanations",
                value=True,
                help="Toggle plain-English explanations for the equations below."
            )

            shown_train_symbols = set()

            def render_inline_symbol_definition(symbol_latex: str, meaning: str, source: str):
                if symbol_latex in shown_train_symbols:
                    return
                shown_train_symbols.add(symbol_latex)
                c1, c2 = st.columns([1, 7])
                c1.latex(symbol_latex)
                c2.markdown(
                    f"<div style='font-size:0.9rem; margin-top:0.1rem; margin-bottom:0.35rem;'>"
                    f"<b>{meaning}</b> <span style='color:#6b7280;'>({source})</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.latex(r"z \sim \mathcal{N}(0, I_{d_z})")
            render_inline_symbol_definition(r"z", "Random noise input vector to generator", f"Sampled each step from Normal distribution with dimension d_z={cfg['noise_dim']}")
            render_inline_symbol_definition(r"\mathcal{N}(0, I_{d_z})", "Noise distribution used to sample z", "Fixed distribution used by CTGAN generator input")
            render_inline_symbol_definition(r"I_{d_z}", "Identity covariance matrix for the noise distribution", f"Dimension implied by d_z={cfg['noise_dim']}")
            render_inline_symbol_definition(r"d_z", "Dimension of noise vector z", f"From saved configuration: {cfg['noise_dim']}")
            if show_math_explanations:
                render_lay_summary("The model starts from random input values.")

            st.latex(rf"G_\theta: \mathbb{{R}}^{{d_z}} \to \mathbb{{R}}^{{d_x}},\quad h_1={g_dim1},\ h_2={g_dim2}")
            render_inline_symbol_definition(r"G_{\theta}", "Generator network mapping noise to synthetic row", f"From configured generator hidden layers: ({g_dim1}, {g_dim2})")
            render_inline_symbol_definition(r"\theta", "Trainable parameters of generator", "Learned during training by optimizer")
            render_inline_symbol_definition(r"d_x", "Dimension of one un-packed feature vector", f"From uploaded dataset column count: {data_dim_train}")
            if show_math_explanations:
                render_lay_summary("The generator turns random input into synthetic rows.")

            st.latex(rf"D_\phi: \mathbb{{R}}^{{d_x}} \to \mathbb{{R}},\quad h_1={d_dim1},\ h_2={d_dim2}")
            render_inline_symbol_definition(r"D_{\phi}", "Discriminator scoring function", f"From configured discriminator hidden layers: ({d_dim1}, {d_dim2})")
            render_inline_symbol_definition(r"\phi", "Trainable parameters of discriminator", "Learned during training by optimizer")
            if show_math_explanations:
                render_lay_summary("The discriminator gives each row a score for how real it looks.")

            cond_enabled_train = cond_col is not None and cond_col in st.session_state.df.columns
            pac_train = cfg['pac']
            cond_term_train = ", c_p" if cond_enabled_train else ""

            if cond_enabled_train:
                cond_card = int(st.session_state.df[cond_col].nunique())
                st.latex(rf"c \in \{{1,\ldots,{cond_card}\}},\quad c_p=\mathrm{{pack}}_p(c),\quad p={pac_train}")
                render_inline_symbol_definition(r"c", "Condition label before packing", f"From condition column '{cond_col}' values")
                render_inline_symbol_definition(r"c_p", "Packed condition vector passed to G and D", "Condition labels grouped into blocks so they align with packed samples")
                render_inline_symbol_definition(r"p", "PAC packing factor", f"From configuration: {pac_train}")
                render_inline_symbol_definition(r"\mathcal{C}", "Set of possible condition labels", f"From unique values in '{cond_col}' (|C|={cond_card})")
                if show_math_explanations:
                    render_lay_summary(f"Because '{cond_col}' is selected, the model uses that label while creating and scoring synthetic rows.")

            st.latex(rf"\tilde{{x}}_p = G_\theta(z{cond_term_train}),\quad z\sim \mathcal{{N}}(0,I_{{d_z}}),\quad x_p=\mathrm{{pack}}_p(x)")
            render_inline_symbol_definition(r"\tilde{x}_p", "Packed synthetic input to discriminator", "Generated by G from z (and packed condition if enabled)")
            render_inline_symbol_definition(r"x_p", "Packed real input to discriminator", f"Real samples grouped by PAC; packed dimension = {data_dim_train * pac_train}")
            render_inline_symbol_definition(r"x", "Real data sample", "Drawn from uploaded dataset")
            if show_math_explanations:
                render_lay_summary("This shows how synthetic rows are created and how real rows are grouped for comparison.")

            st.latex(r"\hat{x}_p = \alpha x_p + (1-\alpha)\tilde{x}_p,\quad \alpha\sim \mathcal{U}(0,1)")
            render_inline_symbol_definition(r"\hat{x}_p", "Packed interpolation used for gradient penalty", "Blend of real and synthetic packed samples")
            render_inline_symbol_definition(r"\alpha", "Interpolation scalar", "Random weight between 0 and 1 for that blend")
            if show_math_explanations:
                render_lay_summary("This creates a blended row between real and synthetic data for the stability check.")

            st.latex(rf"\mathcal{{L}}_D = \mathbb{{E}}[D_\phi(\tilde{{x}}_p{cond_term_train})] - \mathbb{{E}}[D_\phi(x_p{cond_term_train})] + {gp_lambda_train}\,\mathbb{{E}}[(\|\nabla_{{\hat{{x}}_p}}D_\phi(\hat{{x}}_p{cond_term_train})\|_2 - 1)^2]")
            render_inline_symbol_definition(r"\mathcal{L}_D", "Discriminator objective actually minimized", f"Difference between fake and real discriminator scores, plus gradient penalty (λ={gp_lambda_train})")
            render_inline_symbol_definition(r"\mathbb{E}[\cdot]", "Expectation operator", "Average over the current training batch")
            render_inline_symbol_definition(r"\nabla_{\hat{x}_p} D_{\phi}(\hat{x}_p)", "Gradient of discriminator wrt packed interpolated input", "Sensitivity of discriminator output to small input changes")
            render_inline_symbol_definition(r"\|\nabla_{\hat{x}_p} D_{\phi}(\hat{x}_p)\|_2", "Gradient norm in GP term", "Kept close to 1 to stabilize training")
            render_inline_symbol_definition(r"\lambda", "Gradient penalty weight", f"Fixed in model config: {gp_lambda_train}")
            if show_math_explanations:
                render_lay_summary("This is the discriminator's training objective: separate real from synthetic while staying stable.")

            st.latex(rf"\mathcal{{L}}_G = -\mathbb{{E}}[D_\phi(\tilde{{x}}_p{cond_term_train})]")
            render_inline_symbol_definition(r"\mathcal{L}_G", "Generator objective actually minimized", "Encourages synthetic samples to receive more real-like discriminator scores")
            if show_math_explanations:
                render_lay_summary("This is the generator's objective: make synthetic rows score more like real rows.")

            st.latex(rf"\alpha_G={cfg['lr_g']},\quad \alpha_D={cfg['lr_d']},\quad n_{{\mathrm{{discriminator}}}}={cfg['n_critic']},\quad \mathrm{{batch}}={cfg['batch_size']}")
            render_inline_symbol_definition(r"\alpha_G", "Generator learning rate", f"From configuration: {cfg['lr_g']}")
            render_inline_symbol_definition(r"\alpha_D", "Discriminator learning rate", f"From configuration: {cfg['lr_d']}")
            render_inline_symbol_definition(r"n_{\mathrm{discriminator}}", "Number of discriminator updates per generator update", f"From configuration: {cfg['n_critic']}")
            render_inline_symbol_definition(r"\mathrm{batch}", "Mini-batch size per optimization step", f"From configuration: {cfg['batch_size']}")
            if show_math_explanations:
                render_lay_summary("These settings control training speed: learning rates, discriminator updates per round, and batch size.")

            st.latex(rf"\mathrm{{max\_epochs}}={cfg['epochs']},\quad \mathrm{{patience}}={cfg['patience']},\quad \Delta_{{\min}}={cfg['min_delta']}")
            render_inline_symbol_definition(r"\mathrm{max\_epochs}", "Maximum training epochs", f"From configuration: {cfg['epochs']}")
            render_inline_symbol_definition(r"\mathrm{patience}", "Early-stopping patience window", f"From configuration: {cfg['patience']}")
            render_inline_symbol_definition(r"\Delta_{\min}", "Minimum improvement threshold for early stopping", f"From configuration: {cfg['min_delta']}")
            if show_math_explanations:
                render_lay_summary("Training stops when it reaches the epoch limit, or earlier if results stop improving enough.")
        
        # training button
        if st.button("Start Training", help="Begin training the CTGAN model"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            training_visual = st.empty()
            loss_chart = st.empty()
            results_placeholder = st.empty()

            spin_candidates = [
                Path(__file__).parent / "github_site" / "SCOG_pics" / "spin.gif",
                Path(__file__).parent / "SCOG_pics" / "spin.gif"
            ]
            spin_path = next((path for path in spin_candidates if path.exists()), None)
            
            try:
                with st.spinner("Training in progress..."):
                    if spin_path is not None:
                        with open(spin_path, "rb") as spin_file:
                            spin_data_uri = base64.b64encode(spin_file.read()).decode("utf-8")
                        training_visual.markdown(
                            f"<div style='text-align:center; margin: 0.25rem 0 0.6rem 0;'><img src='data:image/gif;base64,{spin_data_uri}' alt='Training in progress' style='width: 220px; height: auto;'></div>",
                            unsafe_allow_html=True
                        )

                    # create model
                    status_text.text("Creating model...")
                    progress_bar.progress(10)

                    model_class = CorrelationAwareCTGAN if st.session_state.config.get('use_correlation_model', False) else CustomCTGAN
                    if st.session_state.config.get('encoding_mode', 'onehot') == 'ordinal':
                        model_class = OrdinalCTGAN if model_class is CustomCTGAN else CorrelationAwareCTGAN

                    selected_ordinal_cols = st.session_state.config.get('ordinal_cols', []) if st.session_state.config.get('encoding_mode', 'onehot') == 'ordinal' else []
                    selected_ordinal_orders = st.session_state.config.get('ordinal_orders', {}) if st.session_state.config.get('encoding_mode', 'onehot') == 'ordinal' else {}

                    categorical_for_model = [c for c in st.session_state.config['categorical_cols'] if c not in selected_ordinal_cols]
                    binary_for_model = [c for c in st.session_state.config['binary_cols'] if c not in selected_ordinal_cols]

                    model_kwargs = dict(
                        continuous_cols=st.session_state.config['continuous_cols'],
                        categorical_cols=categorical_for_model,
                        binary_cols=binary_for_model,
                        ordinal_cols=selected_ordinal_cols,
                        ordinal_orders=selected_ordinal_orders,
                        condition_col=st.session_state.config['condition_col'],
                        noise_dim=st.session_state.config['noise_dim'],
                        generator_dim=st.session_state.config['generator_dim'],
                        discriminator_dim=st.session_state.config['discriminator_dim'],
                        batch_size=st.session_state.config['batch_size'],
                        n_critic=st.session_state.config['n_critic'],
                        lr_g=st.session_state.config['lr_g'],
                        lr_d=st.session_state.config['lr_d'],
                        pac=st.session_state.config['pac'],
                        epochs=st.session_state.config['epochs'],
                        wgan_gp=True,
                        gp_weight=10.0,
                        dropout=0.1,
                        verbose=False,
                        early_stopping=True,
                        patience=st.session_state.config['patience'],
                        min_delta=st.session_state.config['min_delta']
                    )

                    # preflight validation for common failure cases
                    if len(st.session_state.df) < 2:
                        raise ValueError(
                            "Dataset has fewer than 2 rows after preprocessing. "
                            "Please restore data or relax missing/outlier removal settings."
                        )

                    bad_continuous = []
                    for col in model_kwargs['continuous_cols']:
                        if col not in st.session_state.df.columns:
                            bad_continuous.append(col)
                            continue
                        coerced = pd.to_numeric(st.session_state.df[col], errors='coerce')
                        if coerced.notna().sum() == 0:
                            bad_continuous.append(col)
                    if bad_continuous:
                        raise ValueError(
                            "These Continuous columns are not numeric after preprocessing: "
                            + ", ".join(bad_continuous)
                            + ". Reclassify them in Preprocessing (Categorical/Binary/Ordinal)."
                        )

                    if len(model_kwargs['continuous_cols']) + len(model_kwargs['categorical_cols']) + len(model_kwargs['binary_cols']) + len(model_kwargs['ordinal_cols']) == 0:
                        raise ValueError("No feature columns selected for training. Please classify columns in Preprocessing.")

                    if len(st.session_state.df) < model_kwargs['batch_size']:
                        model_kwargs['batch_size'] = max(1, len(st.session_state.df))
                        status_text.text(
                            f"Adjusted batch size to {model_kwargs['batch_size']} (dataset has {len(st.session_state.df)} rows)."
                        )
                        progress_bar.progress(20)

                    if model_kwargs['pac'] > 1:
                        status_text.text(
                            "PAC > 1 is currently disabled for stability in this build; using PAC=1 for training."
                        )
                        model_kwargs['pac'] = 1
                        progress_bar.progress(22)

                    if model_class is CorrelationAwareCTGAN:
                        model_kwargs['correlation_loss_weight'] = st.session_state.config.get('correlation_loss_weight', 1.0)

                    model = model_class(**model_kwargs)
                    
                    # train model
                    status_text.text("Training model... (this may take a few minutes)")
                    progress_bar.progress(25)
                    training_start_time = time.time()
                    epoch_timestamps = {}

                    def on_training_progress(payload: dict):
                        event = payload.get('event')
                        epoch = int(payload.get('epoch', 0))
                        total_epochs = max(1, int(payload.get('total_epochs', st.session_state.config['epochs'])))

                        if event == 'epoch_end':
                            now = time.time()
                            epoch_timestamps[epoch] = now
                            avg_g = payload.get('avg_g_loss', None)
                            avg_d = payload.get('avg_d_loss', None)
                            pct = 25 + int((epoch / total_epochs) * 55)
                            pct = max(25, min(80, pct))
                            progress_bar.progress(pct)

                            elapsed = now - training_start_time
                            avg_epoch_seconds = elapsed / max(1, epoch)
                            remaining_epochs = max(0, total_epochs - epoch)
                            eta_seconds = int(avg_epoch_seconds * remaining_epochs)
                            eta_minutes = eta_seconds // 60
                            eta_rem_seconds = eta_seconds % 60
                            eta_text = f"ETA: {eta_minutes}m {eta_rem_seconds}s"

                            if avg_g is not None and avg_d is not None:
                                status_text.text(
                                    f"Training model... Epoch {epoch}/{total_epochs} | Avg G loss: {avg_g:.4f} | Avg D loss: {avg_d:.4f} | {eta_text}"
                                )
                            else:
                                status_text.text(f"Training model... Epoch {epoch}/{total_epochs} | {eta_text}")

                    model.fit(st.session_state.df, progress_callback=on_training_progress)
                    
                    status_text.text("Training complete!")
                    progress_bar.progress(82)
                    
                    # save model
                    st.session_state.model = model
                    st.session_state.model_trained = True
                    
                    # show training results
                    status_text.text("Generating synthetic data...")
                    progress_bar.progress(88)
                    
                    # generate synthetic data
                    synthetic_df = model.sample(len(st.session_state.df))
                    st.session_state.synthetic_df = synthetic_df
                    
                    # compute metrics
                    status_text.text("Computing metrics...")
                    progress_bar.progress(93)
                    metrics = EvaluationMetrics.summarize_metrics(st.session_state.df, synthetic_df)
                    st.session_state.metrics = metrics

                    corr_report = compute_correlation_report(st.session_state.df, synthetic_df)
                    st.session_state.correlation_report = corr_report
                    
                    status_text.text("Finalizing results...")
                    progress_bar.progress(98)

                    progress_bar.progress(100)
                    status_text.text("Training complete.")
                    training_visual.empty()
                    
                    # display training summary
                    with results_placeholder.container():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Total Epochs",
                                len(model.g_losses) // max(1, len(st.session_state.df) // st.session_state.config['batch_size']),
                                help="Number of epochs trained"
                            )
                        with col2:
                            st.metric(
                                "Total Iterations",
                                len(model.g_losses),
                                help="Number of training iterations"
                            )
                        with col3:
                            st.metric(
                                "Synthetic Rows",
                                len(synthetic_df),
                                help="Generated synthetic rows"
                            )
                        if st.session_state.config.get('use_correlation_model', False):
                            st.caption(
                                f"Model used: CorrelationAwareCTGAN (weight={st.session_state.config.get('correlation_loss_weight', 1.0):.2f})"
                            )
                        elif st.session_state.config.get('encoding_mode', 'onehot') == 'ordinal':
                            st.caption("Model used: OrdinalCTGAN")
                        else:
                            st.caption("Model used: Default CustomCTGAN")
                        st.caption(
                            f"Encoding mode: {st.session_state.config.get('encoding_mode', 'onehot')}"
                        )
                        
                        st.success("Model trained successfully.")
                        st.info("Go to 'View Results' page to see metrics and visualizations")
            
            except Exception as e:
                training_visual.empty()
                st.error(f"Training failed: {str(e)}")
                import traceback
                st.error(traceback.format_exc())


# page 5: view results

elif page == "View Results":
    st.markdown('<div class="section-header">View Results</div>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.error("Please train a model first.")
    else:
        # display metrics
        st.subheader("Evaluation Metrics", anchor=False)
        
        metrics = st.session_state.metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("RMSE", f"{metrics['rmse']:.6f}", help="Root Mean Squared Error")
        with col2:
            st.metric("MSE", f"{metrics['mse']:.6f}", help="Mean Squared Error")
        with col3:
            st.metric("MAE", f"{metrics['mae']:.6f}", help="Mean Absolute Error")
        with col4:
            auc_val = metrics.get('auc', 0.5)
            st.metric("AUC", f"{auc_val:.4f}", help="Real vs Synthetic distinguishability")

        cfg_for_distance = st.session_state.get('config', {})
        cont_for_distance = list(cfg_for_distance.get('continuous_cols', []))
        cat_for_distance = list(dict.fromkeys(
            list(cfg_for_distance.get('categorical_cols', []))
            + list(cfg_for_distance.get('binary_cols', []))
            + list(cfg_for_distance.get('ordinal_cols', []))
        ))

        distance_report = compute_dimension_wise_distance(
            st.session_state.df,
            st.session_state.synthetic_df,
            cont_for_distance,
            cat_for_distance,
        )

        st.subheader("Dimension-wise Distance", anchor=False)
        if distance_report is None:
            st.info("No eligible columns found to compute dimension-wise distance.")
        else:
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.metric("Columns Scored", distance_report['n_columns'])
            with d2:
                st.metric("Composite Sum", f"{distance_report['raw_sum']:.4f}")
            with d3:
                st.metric("Dimension-wise Distance", f"{distance_report['dimension_wise_distance']:.4f}")
            with d4:
                st.metric("Manhattan (L1)", f"{distance_report['l1_sum']:.4f}")

            st.caption(
                f"Normalized Manhattan (L1): {distance_report['l1_normalized']:.4f}"
            )

            st.caption(
                "Continuous columns use normalized Wasserstein distance; categorical/ordinal columns use absolute prevalence difference (total variation)."
            )
            st.dataframe(distance_report['details'], width='stretch')
        
        # display detailed metrics
        st.subheader("Column-wise Analysis", anchor=False)
        
        col_stats = metrics.get('column_stats', {})
        
        if col_stats:
            # categorize columns
            excellent_cols = []
            good_cols = []
            fair_cols = []
            poor_cols = []
            
            for col, stats in col_stats.items():
                mean_diff_pct = (stats['mean_diff'] / (abs(stats['real_mean']) + 1e-8)) * 100
                std_diff_pct = (stats['std_diff'] / (abs(stats['real_std']) + 1e-8)) * 100
                
                col_info = (col, stats, mean_diff_pct, std_diff_pct)
                
                if mean_diff_pct < 5 and std_diff_pct < 10:
                    excellent_cols.append(col_info)
                elif mean_diff_pct < 15 and std_diff_pct < 25:
                    good_cols.append(col_info)
                elif mean_diff_pct < 30 and std_diff_pct < 50:
                    fair_cols.append(col_info)
                else:
                    poor_cols.append(col_info)
            
            # display by category
            if excellent_cols:
                st.markdown("**[EXCELLENT] Highly accurate**")
                for col, stats, mean_pct, std_pct in excellent_cols:
                    with st.expander(f"{col} (Mean Diff: {mean_pct:.1f}%, Std Diff: {std_pct:.1f}%)"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write(f"**Real Data**")
                            st.write(f"Mean: {stats['real_mean']:.4f}")
                            st.write(f"Std: {stats['real_std']:.4f}")
                        with col_b:
                            st.write(f"**Synthetic Data**")
                            st.write(f"Mean: {stats['synth_mean']:.4f}")
                            st.write(f"Std: {stats['synth_std']:.4f}")
            
            if good_cols:
                st.markdown("**[GOOD] Well-matched**")
                for col, stats, mean_pct, std_pct in good_cols:
                    with st.expander(f"{col} (Mean Diff: {mean_pct:.1f}%, Std Diff: {std_pct:.1f}%)"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write(f"**Real Data**")
                            st.write(f"Mean: {stats['real_mean']:.4f}")
                            st.write(f"Std: {stats['real_std']:.4f}")
                        with col_b:
                            st.write(f"**Synthetic Data**")
                            st.write(f"Mean: {stats['synth_mean']:.4f}")
                            st.write(f"Std: {stats['synth_std']:.4f}")
            
            if fair_cols:
                st.markdown("**[FAIR] Moderate differences**")
                for col, stats, mean_pct, std_pct in fair_cols:
                    with st.expander(f"{col} (Mean Diff: {mean_pct:.1f}%, Std Diff: {std_pct:.1f}%)"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write(f"**Real Data**")
                            st.write(f"Mean: {stats['real_mean']:.4f}")
                            st.write(f"Std: {stats['real_std']:.4f}")
                        with col_b:
                            st.write(f"**Synthetic Data**")
                            st.write(f"Mean: {stats['synth_mean']:.4f}")
                            st.write(f"Std: {stats['synth_std']:.4f}")
            
            if poor_cols:
                st.markdown("**[POOR] Significant differences**")
                for col, stats, mean_pct, std_pct in poor_cols:
                    with st.expander(f"{col} (Mean Diff: {mean_pct:.1f}%, Std Diff: {std_pct:.1f}%)", expanded=True):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write(f"**Real Data**")
                            st.write(f"Mean: {stats['real_mean']:.4f}")
                            st.write(f"Std: {stats['real_std']:.4f}")
                        with col_b:
                            st.write(f"**Synthetic Data**")
                            st.write(f"Mean: {stats['synth_mean']:.4f}")
                            st.write(f"Std: {stats['synth_std']:.4f}")
                        st.warning("Warning: consider retraining with more epochs, adjusted hyperparameters, or additional data.")
            
            # summary
            total = len(col_stats)
            excellent = len(excellent_cols)
            good = len(good_cols)
            fair = len(fair_cols)
            poor = len(poor_cols)
            
            st.subheader("Summary", anchor=False)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Excellent", excellent)
            with col2:
                st.metric("Good", good)
            with col3:
                st.metric("Fair", fair)
            with col4:
                st.metric("Poor", poor)
            
            success_rate = ((excellent + good) / total * 100) if total > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # visualizations section
        st.subheader("Visualizations & Analysis", anchor=False)

        corr_report = st.session_state.get('correlation_report')
        correlation_mode_enabled = st.session_state.get('config', {}).get('use_correlation_model', False)
        if correlation_mode_enabled and corr_report is not None and 'mean_abs_diff' in corr_report:
            st.subheader("Correlation Preservation", anchor=False)
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Mean |Δ correlation|", f"{corr_report['mean_abs_diff']:.4f}")
            with c2:
                st.metric("Max |Δ correlation|", f"{corr_report['max_abs_diff']:.4f}")

            st.markdown("**Correlation Heatmaps**")
            h1, h2 = st.columns(2)
            with h1:
                render_correlation_heatmap(corr_report['real_corr'], "Original Data Correlations")
            with h2:
                render_correlation_heatmap(corr_report['synth_corr'], "Synthetic Data Correlations")

            with st.expander("Correlation matrices and absolute differences", expanded=False):
                st.write("Real data correlation (Spearman)")
                st.dataframe(corr_report['real_corr'], width='stretch')
                st.write("Synthetic data correlation (Spearman)")
                st.dataframe(corr_report['synth_corr'], width='stretch')
                st.write("Absolute difference |Synthetic - Real|")
                st.dataframe(corr_report['abs_diff'], width='stretch')

        with st.expander("Training Loss Curves", expanded=False):
            st.write("Generator and discriminator losses over training iterations.")
            try:
                model = st.session_state.model
                ts = datetime.now().strftime('%Y%m%d%H%M%S%f')
                loss_plot_path = os.path.join('.', f'training_losses_{ts}.png')
                EvaluationMetrics.plot_losses(model.g_losses, model.d_losses, loss_plot_path)
                # read and display image bytes to avoid caching
                with open(loss_plot_path, 'rb') as f:
                    img_bytes = f.read()
                st.image(img_bytes, width="stretch")
            except Exception as e:
                st.warning(f"Could not generate loss plot: {e}")
        
        with st.expander("Distribution Comparisons (Numeric Columns)", expanded=False):
            st.write("Histograms comparing real vs synthetic distributions for numeric columns.")
            try:
                ts = datetime.now().strftime('%Y%m%d%H%M%S%f')
                dist_plot_path = os.path.join('.', f'distributions_{ts}.png')
                EvaluationMetrics.plot_column_distributions(st.session_state.df, st.session_state.synthetic_df, dist_plot_path)
                with open(dist_plot_path, 'rb') as f:
                    img_bytes = f.read()
                st.image(img_bytes, width="stretch")
            except Exception as e:
                st.warning(f"Could not generate distribution plot: {e}")
        
        with st.expander("Categorical Distribution Comparisons", expanded=False):
            st.write("Bar charts comparing real vs synthetic distributions for categorical columns.")
            try:
                ts = datetime.now().strftime('%Y%m%d%H%M%S%f')
                cat_plot_path = os.path.join('.', f'categorical_distributions_{ts}.png')
                selected_cat_cols = list(dict.fromkeys((st.session_state.categorical_cols or []) + (st.session_state.binary_cols or [])))
                EvaluationMetrics.plot_categorical_distributions(
                    st.session_state.df,
                    st.session_state.synthetic_df,
                    cat_plot_path,
                    categorical_cols=selected_cat_cols if selected_cat_cols else None,
                )
                if os.path.exists(cat_plot_path):
                    with open(cat_plot_path, 'rb') as f:
                        img_bytes = f.read()
                    st.image(img_bytes, width="stretch")
                else:
                    st.info("No categorical columns found to compare for this dataset.")
            except Exception as e:
                st.warning(f"Could not generate categorical distribution plot: {e}")
        
        with st.expander("PCA Projection (2D)", expanded=False):
            st.write("2D PCA visualization showing how real and synthetic samples cluster together.")
            try:
                ts = datetime.now().strftime('%Y%m%d%H%M%S%f')
                pca_plot_path = os.path.join('.', f'pca_plot_{ts}.png')
                EvaluationMetrics.plot_pca(st.session_state.df, st.session_state.synthetic_df, output_path=pca_plot_path)
                with open(pca_plot_path, 'rb') as f:
                    img_bytes = f.read()
                st.image(img_bytes, width="stretch")
            except Exception as e:
                st.warning(f"Could not generate PCA plot: {e}")
        
        # display sample synthetic data
        st.subheader("Sample Synthetic Data", anchor=False)
        st.dataframe(st.session_state.synthetic_df.head(10), use_container_width=True)
        
        # compute and display k-anonymity for synthetic data
        st.subheader("Privacy Assessment: k-Anonymity", anchor=False)
        with st.expander("k-Anonymity Analysis for Synthetic Data"):
            st.write("**k-Anonymity** measures the minimum number of rows sharing same quasi-identifier values.")
            st.write("This is useful for ensuring individual patients cannot be identified from the dataset.")
            st.write("Higher k = better privacy. k >= 3 is considered reasonably safe.")
            

            k_anon_synthetic = EvaluationMetrics.compute_k_anonymity(st.session_state.synthetic_df)
            k_anon_real = EvaluationMetrics.compute_k_anonymity(st.session_state.df)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                real_k = k_anon_real['k_anonymity']
                synth_k = k_anon_synthetic['k_anonymity']
                if real_k == float('inf'):
                    st.metric("Real Data k-Anonymity", "∞")
                else:
                    st.metric("Real Data k-Anonymity", f"{real_k}")
            with col2:
                if synth_k == float('inf'):
                    st.metric("Synthetic Data k-Anonymity", "∞")
                else:
                    st.metric("Synthetic Data k-Anonymity", f"{synth_k}")
            with col3:
                synth_coverage = k_anon_synthetic['coverage']
                st.metric("Safe Coverage (%)", f"{synth_coverage:.1f}%")
            
            st.info(f"Quasi-identifiers: {', '.join(k_anon_synthetic['quasi_identifiers_used'])}")
        
        # display love plot and smd metrics
        st.subheader("Distribution Balance: Standardized Mean Difference (SMD)", anchor=False)
        with st.expander("Love Plot & SMD Analysis"):
            st.write("**Love plots** visualize covariate balance using Standardized Mean Differences (SMD).")
            st.write("SMD < 0.1 (green) = Good balance | SMD 0.1-0.2 (orange) = Acceptable | SMD > 0.2 (red) = Needs attention")

            # generate and display love plot for full dataset
            try:
                ts = datetime.now().strftime('%Y%m%d%H%M%S%f')
                love_plot_full = os.path.join('.', f'love_plot_full_{ts}.png')
                EvaluationMetrics.plot_love_plot(st.session_state.df, st.session_state.synthetic_df, love_plot_full)
                st.subheader("Love Plot: Full Dataset")
                if os.path.exists(love_plot_full):
                    with open(love_plot_full, 'rb') as f:
                        img_bytes = f.read()
                    st.image(img_bytes, width="stretch")
                else:
                    st.warning("Love plot file not found")
            except Exception as e:
                st.warning(f"Could not generate full Love plot: {e}")

            # display smd values as a table
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("SMD Values")
                if 'smd' in st.session_state.metrics:
                    smd_dict = st.session_state.metrics['smd']
                    smd_data = []
                    for col, smd_val in sorted(smd_dict.items(), key=lambda x: abs(x[1]), reverse=True):
                        status = 'Good' if abs(smd_val) < 0.1 else 'Acceptable' if abs(smd_val) < 0.2 else 'High'
                        smd_data.append({'Feature': col, 'SMD': f'{smd_val:.4f}', 'Status': status})
                    smd_df = pd.DataFrame(smd_data)
                    st.dataframe(smd_df, use_container_width=True)

            with col2:
                st.subheader("SMD Summary")
                if 'smd' in st.session_state.metrics:
                    smd_dict = st.session_state.metrics['smd']
                    good_count = sum(1 for v in smd_dict.values() if abs(v) < 0.1)
                    acceptable_count = sum(1 for v in smd_dict.values() if 0.1 <= abs(v) < 0.2)
                    high_count = sum(1 for v in smd_dict.values() if abs(v) >= 0.2)

                    st.write(f"Good (SMD < 0.1): {good_count} features")
                    st.write(f"Acceptable (0.1 <= SMD < 0.2): {acceptable_count} features")
                    st.write(f"High (SMD >= 0.2): {high_count} features")

            # optional: propensity score matching
            st.markdown("---")
            st.subheader("Propensity Score Matching (PSM)")
            st.write("Run nearest-neighbour PSM to create matched subsets and compare balance before/after matching.")

            psm_enable = st.checkbox("Enable PSM")
            if psm_enable:
                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    match_ratio = st.number_input("Matching ratio (real:synthetic)", min_value=1, max_value=10, value=1, step=1)
                    caliper = st.slider("Caliper (max propensity distance)", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
                with p_col2:
                    feature_select = st.multiselect("Features for PS model (defaults to numeric)", options=st.session_state.df.select_dtypes(include=[np.number]).columns.tolist(), default=None)
                    direction = st.selectbox("Matching Direction", options=["synth_to_real", "real_to_synth"], index=0, help="Choose whether to match synthetic records to real records or vice-versa")
                    seed = st.number_input("Random seed (optional)", min_value=0, max_value=2**31-1, value=0)
                    run_psm = st.button("Run PSM")

                if run_psm:
                    try:
                        with st.spinner("Running propensity score estimation and matching..."):
                            psm_result = EvaluationMetrics.perform_propensity_score_matching(
                                st.session_state.df,
                                st.session_state.synthetic_df,
                                features=feature_select if feature_select else None,
                                ratio=int(match_ratio),
                                caliper=float(caliper),
                                direction=direction,
                                random_state=int(seed) if seed != 0 else None
                            )

                        st.success(f"PSM completed (PS model AUC: {psm_result['auc']:.3f})")

                        matched_real = psm_result['matched_real']
                        matched_synth = psm_result['matched_synthetic']

                        # compute pre/post smd
                        smd_pre = st.session_state.metrics.get('smd', {})
                        smd_post = EvaluationMetrics.compute_standardized_mean_difference(matched_real, matched_synth)

                        # save love plots
                        ts = datetime.now().strftime('%Y%m%d%H%M%S%f')
                        pre_plot = os.path.join('.', f'love_plot_pre_{ts}.png')
                        post_plot = os.path.join('.', f'love_plot_post_{ts}.png')
                        EvaluationMetrics.plot_love_plot(st.session_state.df, st.session_state.synthetic_df, pre_plot)
                        if not matched_real.empty and not matched_synth.empty:
                            EvaluationMetrics.plot_love_plot(matched_real, matched_synth, post_plot)

                        # display plots side-by-side
                        st.markdown('---')
                        img_col1, img_col2 = st.columns(2)
                        with img_col1:
                            st.subheader('Before Matching')
                            if os.path.exists(pre_plot):
                                with open(pre_plot, 'rb') as f:
                                    img_bytes = f.read()
                                st.image(img_bytes, width="stretch")
                            else:
                                st.warning("Pre-matching plot not found")
                            st.write('SMD summary before matching:')
                            st.write({k: float(f"{v:.4f}") for k, v in smd_pre.items()})
                        with img_col2:
                            st.subheader('After Matching')
                            if os.path.exists(post_plot):
                                with open(post_plot, 'rb') as f:
                                    img_bytes = f.read()
                                st.image(img_bytes, width="stretch")
                            else:
                                st.warning("Post-matching plot not found")
                            st.write('SMD summary after matching:')
                            st.write({k: float(f"{v:.4f}") for k, v in smd_post.items()})

                        # store matched samples in session state for further inspection/download
                        st.session_state.matched_real = matched_real
                        st.session_state.matched_synthetic = matched_synth
                        st.session_state.psm_result = psm_result

                        # show small preview of matched sets in the ui
                        st.markdown('---')
                        st.subheader('Matched Samples Preview')
                        preview_col1, preview_col2 = st.columns(2)
                        with preview_col1:
                            st.write(f"Matched real rows: {len(matched_real)}")
                            if not matched_real.empty:
                                st.dataframe(matched_real.head(10), use_container_width=True)
                            else:
                                st.info('No matched real rows to preview')
                        with preview_col2:
                            st.write(f"Matched synthetic rows: {len(matched_synth)}")
                            if not matched_synth.empty:
                                st.dataframe(matched_synth.head(10), use_container_width=True)
                            else:
                                st.info('No matched synthetic rows to preview')

                        st.info('Matched sets saved to session. Use Download page to export matched samples.')
                    except Exception as e:
                        st.error(f'PSM failed: {e}')
                        import traceback
                        st.error(traceback.format_exc())

# page 6: method comparison

elif page == "Method Comparison":
    st.markdown('<div class="section-header">Method Comparison</div>', unsafe_allow_html=True)

    if st.session_state.synthetic_df is None or not st.session_state.model_trained:
        st.error("Please train the GAN model first to compare alternative synthesis methods.")
    elif st.session_state.df is None:
        st.error("Please upload data first.")
    else:
        st.subheader("Compare CTGAN Against Alternative Synthesis Methods", anchor=False)
        st.caption("Run simple baseline tabular synthesizers and compare only the metrics you select.")

        source_option = st.radio(
            "Data source for alternative methods",
            options=["Use current preprocessed data", "Use unedited original data"],
            index=0,
            horizontal=True,
            help="Choose whether baselines are fitted on the active preprocessed dataset or the original uploaded dataset."
        )

        if source_option == "Use unedited original data":
            if st.session_state.original_df is None:
                st.warning("Original data snapshot is not available; using current preprocessed data.")
                base_real_df = st.session_state.df.copy()
            else:
                base_real_df = st.session_state.original_df.copy()
        else:
            base_real_df = st.session_state.df.copy()

        st.caption(f"Comparison source rows: {len(base_real_df)} | columns: {len(base_real_df.columns)}")

        random_seed = st.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=42, step=1)
        n_rows_to_generate = st.number_input(
            "Synthetic rows per method",
            min_value=1,
            max_value=max(1, int(max(len(base_real_df), len(st.session_state.synthetic_df)) * 3)),
            value=max(1, len(base_real_df)),
            step=1,
        )

        method_options = [
            "Random Row Sampling",
            "Independent Column Sampling",
            "Gaussian Copula (simple)",
            "CART Leaf Bootstrap",
            "Bayesian Bootstrap",
            "KMeans Cluster Bootstrap",
            "KDE Sampling",
        ]
        selected_methods = st.multiselect(
            "Alternative synthesis methods",
            options=method_options,
            default=method_options,
            help="Select one or more simple baseline methods to compare with CTGAN."
        )

        metric_options = [
            "RMSE",
            "MSE",
            "MAE",
            "AUC",
            "Dimension-wise Distance",
            "Manhattan (L1)",
            "Mean |SMD|",
            "k-Anonymity",
        ]
        selected_metrics = st.multiselect(
            "Metrics to compare",
            options=metric_options,
            default=["RMSE", "MAE", "AUC", "Dimension-wise Distance", "k-Anonymity"],
            help="Only selected metrics are computed and displayed in this comparison tab."
        )
        st.session_state.comparison_selected_metrics = selected_metrics

        run_comparison = st.button("Run Method Comparison")

        if run_comparison:
            if len(selected_methods) == 0:
                st.error("Please select at least one alternative method.")
            elif len(selected_metrics) == 0:
                st.error("Please select at least one metric to compare.")
            else:
                with st.spinner("Generating alternative synthetic datasets and computing selected metrics..."):
                    seed_int = int(random_seed)
                    n_int = int(n_rows_to_generate)

                    ctgan_df = st.session_state.synthetic_df.copy()
                    ctgan_df = ctgan_df.sample(n=n_int, replace=(len(ctgan_df) < n_int), random_state=seed_int).reset_index(drop=True)

                    if source_option == "Use current preprocessed data":
                        continuous_cols = list(st.session_state.get('config', {}).get('continuous_cols', st.session_state.continuous_cols or []))
                        categorical_cols = list(dict.fromkeys(
                            list(st.session_state.get('config', {}).get('categorical_cols', st.session_state.categorical_cols or []))
                            + list(st.session_state.get('config', {}).get('binary_cols', st.session_state.binary_cols or []))
                            + list(st.session_state.get('config', {}).get('ordinal_cols', st.session_state.ordinal_cols or []))
                        ))
                    else:
                        continuous_cols, categorical_cols = infer_column_groups_for_comparison(base_real_df)

                    method_to_df = {
                        'CTGAN (current model)': ctgan_df
                    }

                    for idx, method_name in enumerate(selected_methods):
                        method_seed = seed_int + idx + 1
                        if method_name == "Random Row Sampling":
                            method_to_df[method_name] = generate_random_row_sample(base_real_df, n_int, random_state=method_seed)
                        elif method_name == "Independent Column Sampling":
                            method_to_df[method_name] = generate_independent_column_sample(base_real_df, n_int, random_state=method_seed)
                        elif method_name == "Gaussian Copula (simple)":
                            method_to_df[method_name] = generate_gaussian_copula_sample(base_real_df, n_int, random_state=method_seed)
                        elif method_name == "CART Leaf Bootstrap":
                            method_to_df[method_name] = generate_cart_leaf_bootstrap(base_real_df, n_int, random_state=method_seed)
                        elif method_name == "Bayesian Bootstrap":
                            method_to_df[method_name] = generate_bayesian_bootstrap_sample(base_real_df, n_int, random_state=method_seed)
                        elif method_name == "KMeans Cluster Bootstrap":
                            method_to_df[method_name] = generate_kmeans_cluster_bootstrap(base_real_df, n_int, random_state=method_seed)
                        elif method_name == "KDE Sampling":
                            method_to_df[method_name] = generate_kde_sample(base_real_df, n_int, random_state=method_seed)

                    rows = []
                    for method_name, synth_df in method_to_df.items():
                        metric_values = compute_selected_comparison_metrics(
                            base_real_df,
                            synth_df,
                            selected_metrics,
                            continuous_cols,
                            categorical_cols,
                        )
                        row = {'Method': method_name}
                        row.update(metric_values)
                        rows.append(row)

                    results_df = pd.DataFrame(rows)
                    st.session_state.comparison_results_df = results_df
                    st.session_state.comparison_synth_data = method_to_df

        if st.session_state.comparison_results_df is not None:
            st.subheader("Comparison Results", anchor=False)
            st.dataframe(st.session_state.comparison_results_df, width='stretch', hide_index=True)

            method_descriptions = {
                "CTGAN (current model)": "GAN-based tabular synthesizer that learns joint feature relationships.",
                "Random Row Sampling": "Resamples full real rows with replacement; preserves row-level combinations from observed data.",
                "Independent Column Sampling": "Samples each column independently from its observed distribution; ignores cross-column relationships.",
                "Gaussian Copula (simple)": "Approximates numeric dependencies via a Gaussian copula and samples categoricals by frequency.",
                "CART Leaf Bootstrap": "Partitions data with a decision tree and resamples records within similar leaf groups.",
                "Bayesian Bootstrap": "Weighted bootstrap where row sampling weights are randomly drawn from a Dirichlet distribution.",
                "KMeans Cluster Bootstrap": "Clusters numeric patterns and resamples rows within clusters to preserve coarse structure.",
                "KDE Sampling": "Fits kernel density to numeric features and samples smooth synthetic numeric values; categoricals by frequency.",
            }

            st.markdown("**Method descriptions (brief):**")
            for method_name in st.session_state.comparison_results_df['Method'].tolist():
                desc = method_descriptions.get(method_name, "Baseline method used for tabular synthesis comparison.")
                st.caption(f"- {method_name}: {desc}")

            st.subheader("Synthetic Data Preview by Method", anchor=False)
            available_methods = list(st.session_state.comparison_synth_data.keys())
            selected_preview_method = st.selectbox("Choose method to preview", options=available_methods)
            st.dataframe(st.session_state.comparison_synth_data[selected_preview_method].head(10), width='stretch')


# page 7: download

elif page == "Download":
    st.markdown('<div class="section-header">Download Results</div>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.error("Please train a model first.")
    else:
        st.subheader("Download Data", anchor=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # download synthetic data
            synthetic_csv = st.session_state.synthetic_df.to_csv(index=False)
            st.download_button(
                label="Synthetic Data (CSV)",
                data=synthetic_csv,
                file_name="synthetic_data.csv",
                mime="text/csv",
                help="Download the generated synthetic dataset"
            )
        
        with col2:
            # download real data (cleaned)
            real_csv = st.session_state.df.to_csv(index=False)
            st.download_button(
                label="Real Data (Cleaned) (CSV)",
                data=real_csv,
                file_name="real_data_cleaned.csv",
                mime="text/csv",
                help="Download the cleaned real dataset used for training"
            )

        # if propensity matching was run, allow download of matched sets
        if 'matched_real' in st.session_state or 'matched_synthetic' in st.session_state:
            st.markdown('---')
            st.write('### Matched Samples (from Propensity Score Matching)')
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                if 'matched_real' in st.session_state and not st.session_state.matched_real.empty:
                    matched_real_csv = st.session_state.matched_real.to_csv(index=False)
                    st.download_button(
                        label="Matched Real Samples",
                        data=matched_real_csv,
                        file_name="matched_real_samples.csv",
                        mime="text/csv",
                        help="Download the real records matched during PSM"
                    )
                else:
                    st.info('No matched real samples available for download')

            with mcol2:
                if 'matched_synthetic' in st.session_state and not st.session_state.matched_synthetic.empty:
                    matched_synth_csv = st.session_state.matched_synthetic.to_csv(index=False)
                    st.download_button(
                        label="Matched Synthetic Samples",
                        data=matched_synth_csv,
                        file_name="matched_synthetic_samples.csv",
                        mime="text/csv",
                        help="Download the synthetic records matched during PSM"
                    )
                else:
                    st.info('No matched synthetic samples available for download')

        # method comparison downloads
        if st.session_state.get('comparison_results_df') is not None:
            st.markdown('---')
            st.write('### Download Alternative Method Data')

            comparison_synth_data = st.session_state.get('comparison_synth_data', {})
            if comparison_synth_data:
                st.markdown("**Download synthetic datasets by method**")
                for method_name, method_df in comparison_synth_data.items():
                    safe_name = ''.join(ch.lower() if ch.isalnum() else '_' for ch in method_name).strip('_')
                    if not safe_name:
                        safe_name = 'method'
                    method_csv = method_df.to_csv(index=False)
                    st.download_button(
                        label=f"{method_name} Data (CSV)",
                        data=method_csv,
                        file_name=f"synthetic_{safe_name}.csv",
                        mime="text/csv",
                        help=f"Download synthetic dataset generated by {method_name}"
                    )
        
        # generate and download comprehensive report
        st.subheader("Download Training Reports", anchor=False)
        
        metrics = st.session_state.metrics
        model = st.session_state.model
        
        final_g_loss = f"{model.g_losses[-1]:.6f}" if model.g_losses else "N/A"
        final_d_loss = f"{model.d_losses[-1]:.6f}" if model.d_losses else "N/A"
        
        report_text = f"""CTGAN TRAINING REPORT - STREAMLIT INTERFACE
================================================================================

TIMESTAMP: {pd.Timestamp.now()}

OVERALL PERFORMANCE:
  RMSE: {metrics['rmse']:.6f}
  MSE:  {metrics['mse']:.6f}
  MAE:  {metrics['mae']:.6f}
  AUC:  {metrics.get('auc', 'N/A')}

TRAINING STATISTICS:
  Final G Loss: {final_g_loss}
  Final D Loss: {final_d_loss}
  Total Iterations: {len(model.g_losses)}

DATA SUMMARY:
  Real Data Rows: {len(st.session_state.df)}
  Synthetic Data Rows: {len(st.session_state.synthetic_df)}
  Total Columns: {len(st.session_state.df.columns)}
  Continuous: {len(st.session_state.config['continuous_cols'])}
  Categorical: {len(st.session_state.config['categorical_cols'])}
  Binary: {len(st.session_state.config['binary_cols'])}

COLUMN-WISE STATISTICS:
"""
        
        for col, stats in metrics.get('column_stats', {}).items():
            report_text += f"\n{col}:\n"
            report_text += f"  Real Mean: {stats['real_mean']:.4f}, Synth Mean: {stats['synth_mean']:.4f}\n"
            report_text += f"  Real Std:  {stats['real_std']:.4f}, Synth Std:  {stats['synth_std']:.4f}\n"

        comparison_df = st.session_state.get('comparison_results_df')
        if comparison_df is not None and not comparison_df.empty:
            report_text += "\n\nMETHOD COMPARISON SUMMARY:\n"
            report_text += "-" * 80 + "\n"
            report_text += comparison_df.to_string(index=False)
            report_text += "\n"

        report_text += "\n\nNOTES:\n"
        report_text += "- For plots and full method-comparison visuals, use the PDF report export.\n"
        report_text += "- Metrics are computed on the current active dataset and generated synthetic outputs.\n"
        
        st.download_button(
            label="Simple Training Report (TXT)",
            data=report_text,
            file_name="training_report.txt",
            mime="text/plain",
            help="Download detailed training report"
        )

        selected_comparison_metrics = st.session_state.get('comparison_selected_metrics', [])
        pdf_bytes = build_pdf_report_bytes(
            real_df=st.session_state.df,
            synth_df=st.session_state.synthetic_df,
            metrics=metrics,
            config=st.session_state.config,
            model=model,
            comparison_df=comparison_df,
            selected_comparison_metrics=selected_comparison_metrics,
        )
        st.download_button(
            label="Full Training Report (PDF)",
            data=pdf_bytes,
            file_name="training_report.pdf",
            mime="application/pdf",
            help="Download a multi-page PDF report with metrics, plots, and method comparisons."
        )
        
        st.success("All downloads ready.")

# page 8: resources

elif page == "Resources":
    st.markdown('<div class="section-header">Real Data Sources</div>', unsafe_allow_html=True)

    st.subheader("Crohn's Disease", anchor=False)
    st.markdown("**Randomised Controlled Trial Original Data**")
    st.markdown("Feagan, B.G. et al. (2016) ‘Ustekinumab as Induction and Maintenance Therapy for Crohn’s Disease’, New England Journal of Medicine, 375(20), pp. 1946–1960. Available at: [https://doi.org/10.1056/NEJMoa1602773](https://doi.org/10.1056/NEJMoa1602773).")

    st.markdown("**Observational Study Original Data**")
    st.markdown("Biemans, V.B.C. et al. (2020) ‘Ustekinumab for Crohn’s Disease: Results of the ICC Registry, a Nationwide Prospective Observational Cohort Study’, Journal of Crohn’s and Colitis, 14(1), pp. 33–45. Available at: [https://doi.org/10.1093/ecco-jcc/jjz119](https://doi.org/10.1093/ecco-jcc/jjz119).")

    st.markdown("**External Original Data**")
    st.markdown("Bello, F. et al. (2024) ‘Long-term real-world data of ustekinumab in Crohn’s disease: the Stockholm ustekinumab study’, Therapeutic Advances in Gastroenterology, 17, p. 17562848241242700. Available at: [https://doi.org/10.1177/17562848241242700](https://doi.org/10.1177/17562848241242700).")

    st.subheader("COVID-19", anchor=False)
    st.markdown("**Randomised Controlled Trial Original Data**")
    st.markdown("Demographic data for coronavirus (COVID-19) testing (England): 28 May to 26 August (no date), GOV.UK. Available at: [https://www.gov.uk/government/publications/demographic-data-for-coronavirus-testing-england-28-may-to-26-august/demographic-data-for-coronavirus-covid-19-testing-england-28-may-to-26-august](https://www.gov.uk/government/publications/demographic-data-for-coronavirus-testing-england-28-may-to-26-august/demographic-data-for-coronavirus-covid-19-testing-england-28-may-to-26-august) (Accessed: 8 March 2024).")
    st.markdown("Yang, Z.-R. et al. (2023) ‘Efficacy of SARS-CoV-2 vaccines and the dose–response relationship with three major antibodies: a systematic review and meta-analysis of randomised controlled trials’, The Lancet Microbe, 4(4), pp. e236–e246. Available at: [https://doi.org/10.1016/S2666-5247(22)00390-1](https://doi.org/10.1016/S2666-5247(22)00390-1).")

    st.markdown("**Observational Study Original Data**")
    st.markdown("Bernal, J.L. et al. (2021) ‘Early effectiveness of COVID-19 vaccination with BNT162b2 mRNA vaccine and ChAdOx1 adenovirus vector vaccine on symptomatic disease, hospitalisations and mortality in older adults in England’, medRxiv, p. 2021.03.01.21252652. Available at: [https://doi.org/10.1101/2021.03.01.21252652](https://doi.org/10.1101/2021.03.01.21252652).")
    st.markdown("Demographic data for coronavirus (COVID-19) testing (England): 28 May to 26 August (no date), GOV.UK. Available at: [https://www.gov.uk/government/publications/demographic-data-for-coronavirus-testing-england-28-may-to-26-august/demographic-data-for-coronavirus-covid-19-testing-england-28-may-to-26-august](https://www.gov.uk/government/publications/demographic-data-for-coronavirus-testing-england-28-may-to-26-august/demographic-data-for-coronavirus-covid-19-testing-england-28-may-to-26-august) (Accessed: 8 March 2024).")

    st.markdown("**External Original Data**")
    st.markdown("Cases in England | Coronavirus in the UK (2023). Available at: [https://coronavirus.data.gov.uk/details/cases?areaType=nation&areaName=England](https://coronavirus.data.gov.uk/details/cases?areaType=nation&areaName=England) (Accessed: 28 February 2024).")
    st.markdown("Demographic data for coronavirus (COVID-19) testing (England): 28 May to 26 August (no date), GOV.UK. Available at: [https://www.gov.uk/government/publications/demographic-data-for-coronavirus-testing-england-28-may-to-26-august/demographic-data-for-coronavirus-covid-19-testing-england-28-may-to-26-august](https://www.gov.uk/government/publications/demographic-data-for-coronavirus-testing-england-28-may-to-26-august/demographic-data-for-coronavirus-covid-19-testing-england-28-may-to-26-august) (Accessed: 8 March 2024).")
    st.markdown("Statistics » COVID-19 vaccinations archive (no date). Available at: [https://www.england.nhs.uk/statistics/statistical-work-areas/covid-19-vaccinations/covid-19-vaccinations-archive/](https://www.england.nhs.uk/statistics/statistical-work-areas/covid-19-vaccinations/covid-19-vaccinations-archive/) (Accessed: 28 February 2024).")

    st.subheader("Simulation Methods", anchor=False)
    st.markdown("Comparing Simulated and Synthetic Data Types - Crohn's (GitHub repository). Available at: [https://github.com/N-cizauskas/Comparing-Simulated-and-Synthetic-Data-Types-Crohns](https://github.com/N-cizauskas/Comparing-Simulated-and-Synthetic-Data-Types-Crohns).")
    st.markdown("Comparing Simulated and Synthetic Data Types - COVID (GitHub repository). Available at: [https://github.com/N-cizauskas/Comparing-Simulated-and-Synthetic-Data-Types-COVID](https://github.com/N-cizauskas/Comparing-Simulated-and-Synthetic-Data-Types-COVID).")

# footer
#st.markdown("---")
#st.write("###documentation")
#st.write("""
#-  See the accompanying documentation files for detailed information
#-  CTGAN: Conditional Tabular GAN for synthetic data generation
#- Statistical validation ensures synthetic data quality
#- Easy-to-use interface for non-technical users
#- """)

#st.write("**developed for Newcastle University - Aim 2 Project**")

# session state note for matched sets
#st.markdown('---')
#st.write('**Note:** If you run Propensity Score Matching (PSM), matched sets are saved to `st.session_state.matched_real` and `st.session_state.matched_synthetic`. You can download them from the Download page or inspect them in the View Results page immediately after matching.')
