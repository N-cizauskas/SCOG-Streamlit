import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy import stats as scipy_stats
from typing import Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

CB_BLUE = '#0072B2'
CB_BROWN = '#8C510A'
CB_BROWN_LIGHT = '#BF812D'


class EvaluationMetrics:
    """Compute and visualize CTGAN evaluation metrics."""

    @staticmethod
    def compute_rmse(real: np.ndarray, synthetic: np.ndarray) -> float:
        """Compute RMSE between real and synthetic data (continuous columns only)."""
        return np.sqrt(np.mean((real - synthetic) ** 2))

    @staticmethod
    def compute_mse(real: np.ndarray, synthetic: np.ndarray) -> float:
        """Compute MSE between real and synthetic data."""
        return np.mean((real - synthetic) ** 2)

    @staticmethod
    def compute_mae(real: np.ndarray, synthetic: np.ndarray) -> float:
        """Compute MAE between real and synthetic data."""
        return np.mean(np.abs(real - synthetic))

    @staticmethod
    def compute_column_statistics(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute column-wise mean and std differences."""
        stats = {}
        for col in real_df.columns:
            if pd.api.types.is_numeric_dtype(real_df[col]):
                real_mean = real_df[col].mean()
                synth_mean = synthetic_df[col].mean()
                real_std = real_df[col].std()
                synth_std = synthetic_df[col].std()
                stats[col] = {
                    'real_mean': real_mean,
                    'synth_mean': synth_mean,
                    'mean_diff': abs(real_mean - synth_mean),
                    'real_std': real_std,
                    'synth_std': synth_std,
                    'std_diff': abs(real_std - synth_std)
                }
        return stats

    @staticmethod
    def plot_losses(g_losses: list, d_losses: list, output_path: str = 'losses.png', figsize: Tuple[int, int] = (14, 5)):
        """Plot generator and discriminator losses over training iterations (separate and combined)."""
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # generator loss only
        axes[0].plot(g_losses, label='Generator Loss', color=CB_BLUE, alpha=0.7, linewidth=1.5)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Generator Loss Over Training')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # discriminator loss only
        axes[1].plot(d_losses, label='Discriminator Loss', color=CB_BROWN, alpha=0.7, linewidth=1.5)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Discriminator Loss Over Training')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # combined plot
        axes[2].plot(g_losses, label='Generator Loss', color=CB_BLUE, alpha=0.7, linewidth=1.5)
        axes[2].plot(d_losses, label='Discriminator Loss', color=CB_BROWN, alpha=0.7, linewidth=1.5)
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Generator vs Discriminator Loss')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Loss plot saved to {output_path}")
        plt.close()

    @staticmethod
    def compute_propensity_scores(real_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                                  features: Optional[list] = None,
                                  regularization: float = 1.0,
                                  max_iter: int = 500) -> Dict:
        """
        Fit a propensity score model distinguishing real vs synthetic records.

        Returns a dict with the fitted model, propensity scores for real and synthetic
        and the combined arrays.
        """
        # prepare feature matrix
        if features is None:
            features = real_df.select_dtypes(include=[np.number]).columns.tolist()
        features = [f for f in features if f in real_df.columns and f in synthetic_df.columns]

        if not features:
            raise ValueError('No numeric features available for propensity score modeling.')

        X_real = real_df[features].fillna(0).values
        X_synth = synthetic_df[features].fillna(0).values

        # standardize features for logistic regression
        scaler = StandardScaler()
        X = np.vstack([X_real, X_synth])
        X_scaled = scaler.fit_transform(X)

        # labels: 1 for real, 0 for synthetic
        y = np.hstack([np.ones(X_real.shape[0]), np.zeros(X_synth.shape[0])])

        # fit logistic regression
        model = LogisticRegression(C=regularization, max_iter=max_iter, solver='lbfgs')
        model.fit(X_scaled, y)

        # compute propensity (probability of being real)
        probs = model.predict_proba(X_scaled)[:, 1]
        probs_real = probs[: X_real.shape[0]]
        probs_synth = probs[X_real.shape[0] :]

        # compute AUC for propensity model
        try:
            auc = roc_auc_score(y, probs)
        except Exception:
            auc = 0.5

        return {
            'model': model,
            'scaler': scaler,
            'features': features,
            'probs_real': probs_real,
            'probs_synth': probs_synth,
            'probs_all': probs,
            'auc': float(auc)
        }

    @staticmethod
    def perform_propensity_score_matching(real_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                                          features: Optional[list] = None,
                                          ratio: int = 1,
                                          caliper: Optional[float] = None,
                                          direction: str = 'synth_to_real',
                                          random_state: Optional[int] = None) -> Dict:
        """
        Perform nearest-neighbour propensity score matching (matching synthetic to real).

        Args:
            real_df, synthetic_df: DataFrames
            features: list of features to use for propensity model (defaults to numeric)
            ratio: number of real matches per synthetic (k)
            caliper: maximum allowed absolute difference in propensity scores

        Returns:
            dict with matched DataFrames and indices and propensity results
        """
        ps_result = EvaluationMetrics.compute_propensity_scores(real_df, synthetic_df, features)
        probs_real = ps_result['probs_real']
        probs_synth = ps_result['probs_synth']

        # optionally set random seed for deterministic sampling/tie-breaking
        if random_state is not None:
            np.random.seed(int(random_state))

        # prepare score arrays depending on matching direction
        # 'synth_to_real' (default): for each synthetic record find nearest real record(s)
        # 'real_to_synth': for each real record find nearest synthetic record(s)
        real_scores = probs_real.reshape(-1, 1)
        synth_scores = probs_synth.reshape(-1, 1)

        if direction == 'synth_to_real':
            nn = NearestNeighbors(n_neighbors=ratio, algorithm='auto').fit(real_scores)
            distances, indices = nn.kneighbors(synth_scores)
            # indices: for each synth -> list of real matches
            synth_indices_iter = enumerate(zip(distances, indices))
            mapping_source = 'synthetic'
        elif direction == 'real_to_synth':
            nn = NearestNeighbors(n_neighbors=ratio, algorithm='auto').fit(synth_scores)
            distances, indices = nn.kneighbors(real_scores)
            # indices: for each real -> list of synthetic matches
            synth_indices_iter = enumerate(zip(distances, indices))
            mapping_source = 'real'
        else:
            raise ValueError("direction must be 'synth_to_real' or 'real_to_synth'")

        matched_real_indices = []
        matched_synth_indices = []

        # build matched index pairs according to direction
        for idx_outer, (dists_row, idx_row) in synth_indices_iter:
            # idx_outer corresponds to synthetic index when synth_to_real, otherwise real index
            # apply caliper if provided
            if caliper is not None:
                valid_pairs = [(d, i) for d, i in zip(dists_row, idx_row) if d <= caliper]
                if not valid_pairs:
                    continue
                chosen = valid_pairs
            else:
                chosen = list(zip(dists_row, idx_row))

            if direction == 'synth_to_real':
                synth_idx = idx_outer
                for d, real_i in chosen:
                    matched_real_indices.append(real_i)
                    matched_synth_indices.append(synth_idx)
            else:  # real_to_synth
                real_idx = idx_outer
                for d, synth_i in chosen:
                    matched_real_indices.append(real_idx)
                    matched_synth_indices.append(synth_i)

        # build matched DataFrames
        if not matched_real_indices:
            matched_real_df = pd.DataFrame(columns=real_df.columns)
            matched_synth_df = pd.DataFrame(columns=synthetic_df.columns)
        else:
            matched_real_df = real_df.iloc[matched_real_indices].reset_index(drop=True)
            matched_synth_df = synthetic_df.iloc[matched_synth_indices].reset_index(drop=True)

        return {
            'ps_model': ps_result['model'],
            'scaler': ps_result['scaler'],
            'features': ps_result['features'],
            'auc': ps_result['auc'],
            'matched_real': matched_real_df,
            'matched_synthetic': matched_synth_df,
            'matched_indices_real': matched_real_indices,
            'matched_indices_synth': matched_synth_indices
        }

    @staticmethod
    def plot_pca(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, n_components: int = 2, 
                  output_path: str = 'pca.png', figsize: Tuple[int, int] = (10, 8)):
        """Plot PCA of real vs synthetic data (numeric columns only, common to both)."""
        # select numeric columns common to both dataframes
        real_numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
        synth_numeric_cols = synthetic_df.select_dtypes(include=[np.number]).columns.tolist()
        common_numeric_cols = [c for c in real_numeric_cols if c in synth_numeric_cols]
        
        if not common_numeric_cols:
            print("No common numeric columns for PCA plot.")
            return

        real_numeric = real_df[common_numeric_cols].values
        synth_numeric = synthetic_df[common_numeric_cols].values

        # standardize
        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(real_numeric)
        synth_scaled = scaler.transform(synth_numeric)

        # pca
        pca = PCA(n_components=n_components)
        real_pca = pca.fit_transform(real_scaled)
        synth_pca = pca.transform(synth_scaled)

        # plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.6, s=30, label='Real', color=CB_BLUE)
        ax.scatter(synth_pca[:, 0], synth_pca[:, 1], alpha=0.6, s=30, label='Synthetic', color=CB_BROWN)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title('PCA: Real vs Synthetic Data')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"PCA plot saved to {output_path}")
        plt.close()

    @staticmethod
    def plot_column_distributions(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, 
                                    output_path: str = 'distributions.png', figsize_per_col: Tuple[int, int] = (4, 3)):
        """Plot histograms of numeric columns comparing real and synthetic data (common columns only)."""
        real_numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
        synth_numeric_cols = synthetic_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in real_numeric_cols if c in synth_numeric_cols]
        
        if not numeric_cols:
            print("No common numeric columns for distribution plot.")
            return
        
        n_cols = len(numeric_cols)
        n_rows = (n_cols + 2) // 3  # 3 columns per row

        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            axes[i].hist(real_df[col], bins=30, alpha=0.6, label='Real', color=CB_BLUE, density=True)
            axes[i].hist(synthetic_df[col], bins=30, alpha=0.6, label='Synthetic', color=CB_BROWN, density=True)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Density')
            axes[i].set_title(f'Distribution: {col}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        # hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Distribution plot saved to {output_path}")
        plt.close()

    @staticmethod
    def plot_categorical_distributions(
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        output_path: str = 'categorical_dist.png',
        categorical_cols: Optional[list] = None,
    ):
        """Plot categorical variable distributions (only for columns present in both dataframes)."""
        if categorical_cols:
            cat_cols = [c for c in categorical_cols if c in real_df.columns and c in synthetic_df.columns]
        else:
            cat_cols = real_df.select_dtypes(include=['object', 'category']).columns.tolist()
            cat_cols = [c for c in cat_cols if c in synthetic_df.columns]

            if not cat_cols:
                # fallback: treat low-cardinality shared columns as categorical
                shared_cols = [c for c in real_df.columns if c in synthetic_df.columns]
                inferred = []
                for col in shared_cols:
                    unique_real = real_df[col].nunique(dropna=True)
                    unique_synth = synthetic_df[col].nunique(dropna=True)
                    if max(unique_real, unique_synth) <= 12:
                        inferred.append(col)
                cat_cols = inferred
        
        if not cat_cols:
            print("No categorical columns to plot.")
            return

        n_cols = len(cat_cols)
        n_rows = (n_cols + 1) // 2

        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
        if n_cols == 1:
            axes = [axes[0]]  # ensure iterable for single subplot
        else:
            axes = axes.flatten()

        for i, col in enumerate(cat_cols):
            real_counts = real_df[col].value_counts(normalize=True)
            synth_counts = synthetic_df[col].value_counts(normalize=True)

            # align categories between real and synthetic
            all_categories = sorted(set(real_counts.index) | set(synth_counts.index))
            real_counts = real_counts.reindex(all_categories, fill_value=0)
            synth_counts = synth_counts.reindex(all_categories, fill_value=0)

            x = np.arange(len(all_categories))
            width = 0.35

            axes[i].bar(x - width / 2, real_counts.values, width, label='Real', color=CB_BLUE, alpha=0.7)
            axes[i].bar(x + width / 2, synth_counts.values, width, label='Synthetic', color=CB_BROWN, alpha=0.7)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Proportion')
            axes[i].set_title(f'Distribution: {col}')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(all_categories, rotation=45)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3, axis='y')

        # hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Categorical distribution plot saved to {output_path}")
        plt.close()

    @staticmethod
    def summarize_metrics(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict:
        """Compute and summarize key metrics with statistical testing, AUC, and SMD."""
        # get common numeric columns (condition col may be missing from synthetic)
        real_numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
        synth_numeric_cols = synthetic_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in real_numeric_cols if c in synth_numeric_cols]
        
        if not numeric_cols:
            print("Warning: No numeric columns found in both dataframes for metrics.")
            return {'rmse': 0.0, 'mse': 0.0, 'mae': 0.0, 'column_stats': {}, 'auc': 0.0, 'smd': {}}

        # extract numeric data
        real_numeric = real_df[numeric_cols].values
        synth_numeric = synthetic_df[numeric_cols].values

        # align shapes
        n_min = min(real_numeric.shape[0], synth_numeric.shape[0])
        real_numeric = real_numeric[:n_min]
        synth_numeric = synth_numeric[:n_min]

        # compute metrics
        rmse = EvaluationMetrics.compute_rmse(real_numeric, synth_numeric)
        mse = EvaluationMetrics.compute_mse(real_numeric, synth_numeric)
        mae = EvaluationMetrics.compute_mae(real_numeric, synth_numeric)
        col_stats = EvaluationMetrics.compute_column_statistics(real_df[numeric_cols], synthetic_df[numeric_cols])
        
        # compute standardized mean differences
        smd_dict = EvaluationMetrics.compute_standardized_mean_difference(real_df, synthetic_df)
        
        # compute AUC for binary classification (real vs synthetic)
        auc = EvaluationMetrics.compute_auc_real_vs_synthetic(real_numeric, synth_numeric)
        
        # add statistical tests and SMD to column stats
        for col in col_stats:
            real_col_data = real_df[col].values
            synth_col_data = synthetic_df[col].values
            
            # determine if column is likely continuous or has few unique values (binary/categorical)
            n_unique_real = len(np.unique(real_col_data))
            n_unique_synth = len(np.unique(synth_col_data))
            
            if n_unique_real <= 10:  # binary or categorical
                # use Mann-Whitney U test (non-parametric)
                stat, p_value = scipy_stats.mannwhitneyu(real_col_data, synth_col_data, alternative='two-sided')
                col_stats[col]['test_type'] = 'Mann-Whitney U'
                col_stats[col]['p_value'] = p_value
            else:  # continuous
                # use Kolmogorov-Smirnov test (distribution test)
                stat, p_value = scipy_stats.ks_2samp(real_col_data, synth_col_data)
                col_stats[col]['test_type'] = 'Kolmogorov-Smirnov'
                col_stats[col]['p_value'] = p_value
            
            # determine if significant at 0.05 level
            col_stats[col]['is_significant'] = p_value < 0.05
            col_stats[col]['significance_level'] = 'p < 0.05' if p_value < 0.05 else 'p >= 0.05'
            
            # add SMD
            if col in smd_dict:
                col_stats[col]['smd'] = smd_dict[col]
                col_stats[col]['smd_interpretation'] = 'Good' if abs(smd_dict[col]) < 0.1 else 'Acceptable' if abs(smd_dict[col]) < 0.2 else 'High'

        return {
            'rmse': rmse,
            'mse': mse,
            'mae': mae,
            'column_stats': col_stats,
            'auc': auc,
            'smd': smd_dict
        }
    
    @staticmethod
    def compute_auc_real_vs_synthetic(real_data: np.ndarray, synth_data: np.ndarray) -> float:
        """Compute AUC for classifying real vs synthetic data.
        Higher AUC (closer to 1.0) means synthetic data is distinguishable (worse).
        Lower AUC (closer to 0.5) means synthetic data is similar to real (better).
        """
        try:
            # create binary labels: 1 for real, 0 for synthetic
            X = np.vstack([real_data, synth_data])
            y = np.hstack([np.ones(len(real_data)), np.zeros(len(synth_data))])

            # ensure 2D shape and finite values
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # train/test split for a less biased distinguishability estimate
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.30,
                random_state=42,
                stratify=y,
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf = LogisticRegression(max_iter=1000, solver='lbfgs')
            clf.fit(X_train_scaled, y_train)
            scores = clf.predict_proba(X_test_scaled)[:, 1]

            auc = float(roc_auc_score(y_test, scores))

            # orientation safeguard: we care about separability magnitude
            if auc < 0.5:
                auc = 1.0 - auc

            return auc
        except Exception as e:
            try:
                # fallback: centroid similarity score (real-like = higher score)
                X = np.vstack([real_data, synth_data])
                y = np.hstack([np.ones(len(real_data)), np.zeros(len(synth_data))])
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

                real_mean = np.mean(np.nan_to_num(real_data, nan=0.0, posinf=0.0, neginf=0.0), axis=0)
                distances = np.linalg.norm(X - real_mean, axis=1)
                similarity = -distances
                auc = float(roc_auc_score(y, similarity))
                if auc < 0.5:
                    auc = 1.0 - auc
                return auc
            except Exception as inner_e:
                print(f"Warning: Could not compute AUC: {e}; fallback failed: {inner_e}")
                return 0.5

    @staticmethod
    def print_summary(metrics: Dict):
        """Pretty-print metrics summary with statistical significance and performance highlighting."""
        print("\n" + "=" * 80)
        print("CTGAN EVALUATION METRICS - COMPREHENSIVE REPORT")
        print("=" * 80)
        
        # overall metrics
        print("\nOVERALL PERFORMANCE:")
        print("-" * 80)
        print(f"  RMSE (Root Mean Squared Error):  {metrics['rmse']:.6f}")
        print(f"  MSE  (Mean Squared Error):       {metrics['mse']:.6f}")
        print(f"  MAE  (Mean Absolute Error):      {metrics['mae']:.6f}")
        print(f"  AUC  (Real vs Synthetic):        {metrics['auc']:.6f}  (0.5=identical, 1.0=completely different)")
        
        # AUC interpretation
        if metrics['auc'] < 0.55:
            interpretation = "[EXCELLENT] Synthetic data is nearly indistinguishable"
        elif metrics['auc'] < 0.65:
            interpretation = "[GOOD] Synthetic data is similar to real data"
        elif metrics['auc'] < 0.75:
            interpretation = "[FAIR] Some differences between real and synthetic"
        else:
            interpretation = "[POOR] Synthetic data differs significantly from real"
        print(f"      {interpretation}")
        
        # column-wise analysis
        print("\n" + "=" * 80)
        print("COLUMN-WISE STATISTICAL ANALYSIS")
        print("=" * 80)
        
        col_stats = metrics['column_stats']
        
        # categorize columns by performance
        excellent_cols = []
        good_cols = []
        fair_cols = []
        poor_cols = []
        
        for col, stats in col_stats.items():
            mean_diff_pct = (stats['mean_diff'] / (abs(stats['real_mean']) + 1e-8)) * 100
            std_diff_pct = (stats['std_diff'] / (abs(stats['real_std']) + 1e-8)) * 100
            
            if mean_diff_pct < 5 and std_diff_pct < 10:
                excellent_cols.append((col, stats, mean_diff_pct, std_diff_pct))
            elif mean_diff_pct < 15 and std_diff_pct < 25:
                good_cols.append((col, stats, mean_diff_pct, std_diff_pct))
            elif mean_diff_pct < 30 and std_diff_pct < 50:
                fair_cols.append((col, stats, mean_diff_pct, std_diff_pct))
            else:
                poor_cols.append((col, stats, mean_diff_pct, std_diff_pct))
        
        # print excellent columns
        if excellent_cols:
            print("\n[EXCELLENT] - Highly accurate synthetic distributions:")
            print("-" * 80)
            for col, stats, mean_pct, std_pct in excellent_cols:
                smd_val = stats.get('smd', 0)
                smd_interp = stats.get('smd_interpretation', 'Unknown')
                print(f"\n  {col} (Mean Diff: {mean_pct:.1f}%, Std Diff: {std_pct:.1f}%)")
                print(f"    Real:       Mean={stats['real_mean']:.4f}, Std={stats['real_std']:.4f}")
                print(f"    Synthetic:  Mean={stats['synth_mean']:.4f}, Std={stats['synth_std']:.4f}")
                print(f"    SMD: {smd_val:.4f} ({smd_interp})")
                print(f"    Test: {stats['test_type']} | {stats['significance_level']}")
        
        # print good columns
        if good_cols:
            print("\n[GOOD] - Well-matched synthetic distributions:")
            print("-" * 80)
            for col, stats, mean_pct, std_pct in good_cols:
                smd_val = stats.get('smd', 0)
                smd_interp = stats.get('smd_interpretation', 'Unknown')
                print(f"\n  {col} (Mean Diff: {mean_pct:.1f}%, Std Diff: {std_pct:.1f}%)")
                print(f"    Real:       Mean={stats['real_mean']:.4f}, Std={stats['real_std']:.4f}")
                print(f"    Synthetic:  Mean={stats['synth_mean']:.4f}, Std={stats['synth_std']:.4f}")
                print(f"    SMD: {smd_val:.4f} ({smd_interp})")
                print(f"    Test: {stats['test_type']} | {stats['significance_level']}")
        
        # print fair columns
        if fair_cols:
            print("\n[FAIR] - Moderate differences (may need attention):")
            print("-" * 80)
            for col, stats, mean_pct, std_pct in fair_cols:
                smd_val = stats.get('smd', 0)
                smd_interp = stats.get('smd_interpretation', 'Unknown')
                print(f"\n  {col} (Mean Diff: {mean_pct:.1f}%, Std Diff: {std_pct:.1f}%)")
                print(f"    Real:       Mean={stats['real_mean']:.4f}, Std={stats['real_std']:.4f}")
                print(f"    Synthetic:  Mean={stats['synth_mean']:.4f}, Std={stats['synth_std']:.4f}")
                print(f"    SMD: {smd_val:.4f} ({smd_interp})")
                print(f"    Test: {stats['test_type']} | {stats['significance_level']}")
        
        # print poor columns (highlighted as problematic)
        if poor_cols:
            print("\n[POOR] - SIGNIFICANT DIFFERENCES (Requires investigation):")
            print("=" * 80)
            for col, stats, mean_pct, std_pct in poor_cols:
                smd_val = stats.get('smd', 0)
                smd_interp = stats.get('smd_interpretation', 'Unknown')
                print(f"\n  >>> {col} (Mean Diff: {mean_pct:.1f}%, Std Diff: {std_pct:.1f}%) <<<")
                print(f"    Real:       Mean={stats['real_mean']:.4f}, Std={stats['real_std']:.4f}")
                print(f"    Synthetic:  Mean={stats['synth_mean']:.4f}, Std={stats['synth_std']:.4f}")
                print(f"    SMD: {smd_val:.4f} ({smd_interp})")
                print(f"    Test: {stats['test_type']} | {stats['significance_level']}")
                print(f"    >>> Recommendation: Check data preprocessing, model capacity, or training duration")
        
        # summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        total = len(col_stats)
        excellent = len(excellent_cols)
        good = len(good_cols)
        fair = len(fair_cols)
        poor = len(poor_cols)
        
        print(f"\nColumns: {excellent} excellent, {good} good, {fair} fair, {poor} poor (out of {total} total)")
        print(f"Success Rate: {((excellent + good) / total * 100):.1f}% of columns match well")
        
        if poor > 0:
            print(f"\n[WARNING] {poor} column(s) with poor performance - review recommendations above")
        
        print("=" * 80 + "\n")

    @staticmethod
    def compute_k_anonymity(df: pd.DataFrame, quasi_identifiers: Optional[list] = None) -> Dict[str, any]:
        """
        Compute k-anonymity for a dataset.
        
        k-anonymity: minimum number of rows that share the same combination of quasi-identifiers.
        Higher k-anonymity = better privacy protection (harder to re-identify individuals).
        
        Args:
            df: Input dataframe
            quasi_identifiers: List of column names to use for k-anonymity computation.
                               If None, uses all categorical and binary columns.
        
        Returns:
            Dictionary with k_anonymity, vulnerable_rows, vulnerable_count, coverage
        """
        if quasi_identifiers is None:
            # use categorical and binary columns
            quasi_identifiers = df.select_dtypes(include=['object', 'category']).columns.tolist()
            # also include low-cardinality numeric columns (< 50 unique values)
            for col in df.select_dtypes(include=['int64', 'int32', 'float64', 'float32']).columns:
                if df[col].nunique() < 50:
                    quasi_identifiers.append(col)
        
        if not quasi_identifiers or all(col not in df.columns for col in quasi_identifiers):
            return {
                'k_anonymity': float('inf'),
                'vulnerable_rows': 0,
                'vulnerable_count': 0,
                'coverage': 100.0,
                'average_group_size': len(df),
                'quasi_identifiers_used': quasi_identifiers
            }
        
        # filter to valid columns
        valid_cols = [col for col in quasi_identifiers if col in df.columns]
        
        if not valid_cols:
            return {
                'k_anonymity': float('inf'),
                'vulnerable_rows': 0,
                'vulnerable_count': 0,
                'coverage': 100.0,
                'average_group_size': len(df),
                'quasi_identifiers_used': valid_cols
            }
        
        # group by quasi-identifiers and count occurrences
        group_counts = df[valid_cols].value_counts()
        
        # k-anonymity is the minimum group size
        k_anonymity = group_counts.min()
        
        # count rows that appear k < 3 (more vulnerable)
        vulnerable_counts = (group_counts < 3)
        vulnerable_rows = vulnerable_counts.sum()
        vulnerable_count = 0
        for count in group_counts:
            if count < 3:
                vulnerable_count += count
        
        # coverage: percentage of rows in groups with k >= 3
        safe_count = (group_counts >= 3).sum()
        coverage = (safe_count / len(group_counts) * 100) if len(group_counts) > 0 else 0
        
        average_group_size = group_counts.mean()
        
        return {
            'k_anonymity': int(k_anonymity),
            'vulnerable_rows': int(vulnerable_rows),
            'vulnerable_count': int(vulnerable_count),
            'coverage': float(coverage),
            'average_group_size': float(average_group_size),
            'quasi_identifiers_used': valid_cols
        }

    @staticmethod
    def compute_standardized_mean_difference(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute Standardized Mean Difference (SMD) for each numeric column.
        
        SMD = (mean_synthetic - mean_real) / sqrt((std_real^2 + std_synthetic^2) / 2)
        
        SMD < 0.1 is considered negligible/good balance
        SMD 0.1-0.2 is considered small
        SMD > 0.2 is considered meaningful/concerning
        
        Args:
            real_df: Real dataset (DataFrame)
            synthetic_df: Synthetic dataset (DataFrame)
        
        Returns:
            Dictionary with SMD for each numeric column
        """
        smd_dict = {}
        
        for col in real_df.select_dtypes(include=[np.number]).columns:
            if col not in synthetic_df.columns:
                continue
            
            real_mean = real_df[col].mean()
            synth_mean = synthetic_df[col].mean()
            real_std = real_df[col].std()
            synth_std = synthetic_df[col].std()
            
            # avoid division by zero
            pooled_std = np.sqrt((real_std**2 + synth_std**2) / 2)
            if pooled_std < 1e-10:
                smd = 0.0
            else:
                smd = (synth_mean - real_mean) / pooled_std
            
            smd_dict[col] = float(smd)
        
        return smd_dict

    @staticmethod
    def plot_love_plot(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, 
                       output_path: str = 'love_plot.png', figsize: Tuple[int, int] = (12, 6)):
        """
        Create a Love plot showing Standardized Mean Differences.
        
        A Love plot visualizes the balance of covariates before and after matching.
        For synthetic data, it shows how well features are preserved.
        
        Reference lines:
        - 0.1 (vertical red line): Good balance threshold
        - 0.2 (vertical orange line): Acceptable balance threshold
        
        Args:
            real_df: Real dataset
            synthetic_df: Synthetic dataset
            output_path: Where to save the plot
            figsize: Figure size (width, height)
        """
        smd_dict = EvaluationMetrics.compute_standardized_mean_difference(real_df, synthetic_df)
        
        if not smd_dict:
            print("No numeric columns found for Love plot")
            return
        
        # sort by absolute SMD for better visualization
        sorted_cols = sorted(smd_dict.items(), key=lambda x: abs(x[1]))
        columns = [item[0] for item in sorted_cols]
        smd_values = [item[1] for item in sorted_cols]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # color points based on SMD magnitude
        colors = [CB_BLUE if abs(smd) < 0.1 else CB_BROWN_LIGHT if abs(smd) < 0.2 else CB_BROWN
                  for smd in smd_values]
        
        # plot points
        ax.scatter(smd_values, range(len(columns)), c=colors, s=100, alpha=0.7, zorder=3)
        
        # add reference lines
        ax.axvline(x=0.1, color=CB_BLUE, linestyle='--', linewidth=2, label='Good balance (0.1)', alpha=0.7)
        ax.axvline(x=-0.1, color=CB_BLUE, linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(x=0.2, color=CB_BROWN, linestyle=':', linewidth=2, label='Acceptable (0.2)', alpha=0.7)
        ax.axvline(x=-0.2, color=CB_BROWN, linestyle=':', linewidth=2, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        
        # formatting
        ax.set_yticks(range(len(columns)))
        ax.set_yticklabels(columns)
        ax.set_xlabel('Standardized Mean Difference (SMD)', fontsize=12, fontweight='bold')
        ax.set_title('Love Plot: Standardized Mean Differences\n(Real vs Synthetic Data)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend(loc='best', fontsize=10)
        
        # add text annotations
        ax.text(0.1, ax.get_ylim()[1] * 0.95, 'Good', fontsize=9, color=CB_BLUE, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Love plot saved to {output_path}")
        plt.close()

    @staticmethod
    def plot_smd_boxplots(real_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                          output_path: str = 'smd_boxplots.png', figsize: Tuple[int, int] = (14, 6)):
        """
        Create boxplots showing distribution comparisons highlighting SMD.
        
        Creates side-by-side boxplots for numeric columns.
        
        Args:
            real_df: Real dataset
            synthetic_df: Synthetic dataset
            output_path: Where to save the plot
            figsize: Figure size (width, height)
        """
        numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col in synthetic_df.columns]
        
        if not numeric_cols:
            print("No numeric columns found for boxplot")
            return
        
        # limit to first 6 columns for readability
        numeric_cols = numeric_cols[:6]
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        smd_dict = EvaluationMetrics.compute_standardized_mean_difference(real_df, synthetic_df)
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            
            # create boxplots
            data_to_plot = [real_df[col].dropna(), synthetic_df[col].dropna()]
            bp = ax.boxplot(data_to_plot, labels=['Real', 'Synthetic'], patch_artist=True)
            
            # color boxes
            colors = ['lightblue', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            # add SMD information
            smd_val = smd_dict.get(col, 0)
            smd_status = 'Good' if abs(smd_val) < 0.1 else 'Acceptable' if abs(smd_val) < 0.2 else 'High'
            
            ax.set_title(f'{col}\nSMD: {smd_val:.3f} ({smd_status})', fontweight='bold')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3, axis='y')
        
        # hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"SMD boxplots saved to {output_path}")
        plt.close()
