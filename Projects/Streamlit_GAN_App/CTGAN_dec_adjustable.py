import os
import math
import json
from typing import List, Optional, Dict, Tuple, Any, Callable
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def weights_init(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class TabularDataset(Dataset):
    """Simple dataset for preprocessed tabular numpy arrays."""

    def __init__(self, X: np.ndarray, cond: Optional[np.ndarray] = None):
        self.X = X.astype(np.float32)
        self.cond = None if cond is None else cond.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.cond is None:
            return self.X[idx]
        return self.X[idx], self.cond[idx]


# ----------------------------- Models -----------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims: List[int], dropout=0.0, spectral_norm=False):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims)-1):
            linear = nn.Linear(dims[i], dims[i+1])
            if spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers.append(linear)
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.net.apply(weights_init)

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, noise_dim, cond_dim, output_dim, hidden_dims=(256,256), dropout=0.0, spectral_norm=False):
        super().__init__()
        self.input_dim = noise_dim + (cond_dim or 0)
        self.mlp = MLP(self.input_dim, list(hidden_dims), dropout=dropout, spectral_norm=spectral_norm)
        self.out = nn.Linear(list(hidden_dims)[-1], output_dim)
        self.out.apply(weights_init)

    def forward(self, z, c=None):
        if c is not None:
            x = torch.cat([z, c], dim=1)
        else:
            x = z
        h = self.mlp(x)
        return self.out(h)


class Discriminator(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dims=(256,256), dropout=0.0, spectral_norm=False, aux_classes: Optional[int]=None):
        super().__init__()
        # if conditioning, the discriminator receives concatenated condition
        self.input_dim = input_dim + (cond_dim or 0)
        self.mlp = MLP(self.input_dim, list(hidden_dims), dropout=dropout, spectral_norm=spectral_norm)
        self.real_fake = nn.Linear(list(hidden_dims)[-1], 1)
        self.real_fake.apply(weights_init)
        self.aux_classes = aux_classes
        if aux_classes is not None:
            # auxiliary classifier head for info loss (predict condition labels)
            self.aux = nn.Linear(list(hidden_dims)[-1], aux_classes)
            self.aux.apply(weights_init)

    def forward(self, x, c=None):
        if c is not None:
            x = torch.cat([x, c], dim=1)
        h = self.mlp(x)
        rf = self.real_fake(h)
        if self.aux_classes is not None:
            aux = self.aux(h)
            return rf, aux
        return rf


# ----------------------------- Preprocessing ----------------------------

class SimplePreprocessor:
    """Preprocess a pandas DataFrame into numeric vectors suitable for CTGAN.
    - continuous_cols: left as-is (float)
    - categorical_cols: label-encoded and then one-hot expanded into separate columns
    - binary_cols: treated as categorical with 2 values

    The preprocessor stores mappings to inverse-transform synthetic output back to pandas.
    """

    def __init__(
        self,
        continuous_cols: List[str],
        categorical_cols: List[str],
        binary_cols: List[str]=None,
        ordinal_cols: List[str]=None,
        ordinal_orders: Optional[Dict[str, List[Any]]] = None,
    ):
        self.continuous_cols = continuous_cols or []
        self.categorical_cols = categorical_cols or []
        self.binary_cols = binary_cols or []
        self.ordinal_cols = ordinal_cols or []
        self.ordinal_orders = ordinal_orders or {}

        # internal
        self.cat_categories: Dict[str, List[Any]] = {}
        self.ordinal_categories: Dict[str, List[Any]] = {}
        self.col_order: List[str] = []
        self.col_slices: Dict[str, slice] = {}

    def fit(self, df: pd.DataFrame):
        # register categories
        for c in self.categorical_cols + self.binary_cols:
            cats = list(pd.Categorical(df[c]).categories)
            self.cat_categories[c] = cats

        for c in self.ordinal_cols:
            detected = list(pd.Series(df[c]).dropna().unique())
            requested = self.ordinal_orders.get(c)
            if requested:
                requested_set = {str(v) for v in requested}
                detected_set = {str(v) for v in detected}
                if requested_set == detected_set:
                    cats = list(requested)
                else:
                    cats = detected
            else:
                cats = detected
            self.ordinal_categories[c] = cats

        # determine positions
        idx = 0
        for c in self.continuous_cols:
            self.col_order.append(c)
            self.col_slices[c] = slice(idx, idx+1)
            idx += 1

        for c in self.ordinal_cols:
            self.col_order.append(c)
            self.col_slices[c] = slice(idx, idx+1)
            idx += 1

        for c in self.categorical_cols + self.binary_cols:
            k = len(self.cat_categories[c])
            self.col_order.append(c)
            self.col_slices[c] = slice(idx, idx+k)
            idx += k

        self._vector_dim = idx
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        X = np.zeros((n, self._vector_dim), dtype=np.float32)
        for c in self.continuous_cols:
            X[:, self.col_slices[c]] = df[c].astype(float).values.reshape(-1,1)
        for c in self.ordinal_cols:
            cats = self.ordinal_categories[c]
            arr = pd.Categorical(df[c], categories=cats).codes.astype(float)
            arr[arr < 0] = 0
            X[:, self.col_slices[c]] = arr.reshape(-1, 1)
        for c in self.categorical_cols + self.binary_cols:
            cats = self.cat_categories[c]
            arr = pd.Categorical(df[c], categories=cats).codes
            # one-hot
            for i, cat in enumerate(cats):
                X[:, self.col_slices[c].start + i] = (arr == i).astype(float)
        return X

    def inverse_transform(self, X: np.ndarray) -> pd.DataFrame:
        rows = []
        for r in X:
            row = {}
            for c in self.continuous_cols:
                row[c] = float(r[self.col_slices[c].start])
            for c in self.ordinal_cols:
                sl = self.col_slices[c]
                raw_idx = r[sl.start]
                idx = int(np.round(raw_idx))
                cats = self.ordinal_categories[c]
                if len(cats) == 0:
                    row[c] = None
                else:
                    idx = max(0, min(idx, len(cats) - 1))
                    row[c] = cats[idx]
            for c in self.categorical_cols + self.binary_cols:
                sl = self.col_slices[c]
                onehot = r[sl]
                idx = int(np.argmax(onehot))
                cats = self.cat_categories[c]
                # safety
                if idx >= len(cats):
                    idx = len(cats)-1
                row[c] = cats[idx]
            rows.append(row)
        return pd.DataFrame(rows)

    @property
    def output_dim(self):
        return self._vector_dim


# ----------------------------- Loss helpers -----------------------------

def gradient_penalty(discriminator, real, fake, device, c=None):
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real)
    interpolates = alpha * real + (1 - alpha) * fake
    interpolates.requires_grad_(True)
    if c is not None:
        disc_interpolates = discriminator(interpolates, c)
        # disc_interpolates may be tuple (rf, aux)
        if isinstance(disc_interpolates, tuple):
            disc_interpolates = disc_interpolates[0]
    else:
        disc_interpolates = discriminator(interpolates)
        if isinstance(disc_interpolates, tuple):
            disc_interpolates = disc_interpolates[0]
    grad_outputs = torch.ones_like(disc_interpolates)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    gp = ((grad_norm - 1) ** 2).mean()
    return gp


# ----------------------------- CTGAN class -------------------------------

class CustomCTGAN:
    def __init__(
        self,
        continuous_cols: List[str],
        categorical_cols: List[str],
        binary_cols: List[str] = None,
        ordinal_cols: List[str] = None,
        ordinal_orders: Optional[Dict[str, List[Any]]] = None,
        condition_col: Optional[str] = None,
        noise_dim: int = 128,
        generator_dim: Tuple[int, ...] = (256,256),
        discriminator_dim: Tuple[int, ...] = (256,256),
        embedding_dim: int = 128,
        batch_size: int = 500,
        n_critic: int = 5,
        pac: int = 10,
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
        weight_decay: float = 0.0,
        epochs: int = 200,
        device: Optional[str] = None,
        wgan_gp: bool = True,
        gp_weight: float = 10.0,
        hinge_loss: bool = False,
        dropout: float = 0.0,
        spectral_norm: bool = False,
        aux_info_loss: bool = False,
        aux_loss_weight: float = 1.0,
        verbose: bool = True,
        early_stopping: bool = True,
        patience: int = 3,
        min_delta: float = 1e-4
    ):
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols or []
        self.binary_cols = binary_cols or []
        self.ordinal_cols = ordinal_cols or []
        self.ordinal_orders = ordinal_orders or {}
        self.condition_col = condition_col
        self.noise_dim = noise_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.n_critic = n_critic
        self.pac = pac
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.wgan_gp = wgan_gp
        self.gp_weight = gp_weight
        self.hinge_loss = hinge_loss
        self.dropout = dropout
        self.spectral_norm = spectral_norm
        self.aux_info_loss = aux_info_loss
        self.aux_loss_weight = aux_loss_weight
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.patience = patience  # number of epochs with no improvement before stopping
        self.min_delta = min_delta  # minimum loss change to qualify as improvement
        self.correlation_preservation = False
        self.correlation_loss_weight = 0.0

        # placeholders
        self.preprocessor: Optional[SimplePreprocessor] = None
        self.G: Optional[Generator] = None
        self.D: Optional[Discriminator] = None
        self.optim_G = None
        self.optim_D = None
        self.aux_criterion = None
        # stored condition dimension used at sampling time
        self._cond_dim = 0
        # loss tracking for visualization
        self.g_losses: List[float] = []
        self.d_losses: List[float] = []

    def _correlation_matrix(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(0) < 2:
            size = x.size(1) if x.dim() == 2 else 1
            return torch.zeros((size, size), device=x.device)
        centered = x - x.mean(dim=0, keepdim=True)
        cov = centered.T @ centered / max(1, x.size(0) - 1)
        std = torch.sqrt(torch.clamp(torch.diag(cov), min=1e-8))
        denom = std[:, None] * std[None, :]
        corr = cov / torch.clamp(denom, min=1e-8)
        return torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    def _correlation_preservation_loss(self, real_batch: torch.Tensor, fake_batch: torch.Tensor) -> torch.Tensor:
        if real_batch.size(0) < 4 or fake_batch.size(0) < 4:
            return torch.tensor(0.0, device=fake_batch.device)
        real_corr = self._correlation_matrix(real_batch)
        fake_corr = self._correlation_matrix(fake_batch)
        return F.mse_loss(fake_corr, real_corr)

    def _build(self, input_dim: int, cond_dim: int, aux_classes: Optional[int]):
        self.G = Generator(
            noise_dim=self.noise_dim,
            cond_dim=cond_dim,
            output_dim=input_dim,
            hidden_dims=self.generator_dim,
            dropout=self.dropout,
            spectral_norm=self.spectral_norm
        ).to(self.device)

        self.D = Discriminator(
            input_dim=input_dim,
            cond_dim=cond_dim,
            hidden_dims=self.discriminator_dim,
            dropout=self.dropout,
            spectral_norm=self.spectral_norm,
            aux_classes=aux_classes if self.aux_info_loss else None
        ).to(self.device)

        self.optim_G = torch.optim.Adam(self.G.parameters(), lr=self.lr_g, weight_decay=self.weight_decay, betas=(0.5, 0.9))
        self.optim_D = torch.optim.Adam(self.D.parameters(), lr=self.lr_d, weight_decay=self.weight_decay, betas=(0.5, 0.9))

        if self.aux_info_loss:
            # auxiliary classifier uses cross-entropy
            self.aux_criterion = nn.CrossEntropyLoss()

        # remember condition vector size for sampling
        self._cond_dim = cond_dim or 0

    def fit(self, df: pd.DataFrame, progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        def safe_progress_call(payload: Dict[str, Any]):
            if progress_callback is None:
                return
            try:
                progress_callback(payload)
            except Exception:
                return

        # build preprocessor and dataset
        df = df.copy().reset_index(drop=True)
        feature_df = df.copy()
        cond_values = None
        cond_cats = None
        if self.condition_col is not None:
            if self.condition_col not in df.columns:
                raise ValueError('condition_col not found in dataframe columns')
            cond_series = df[self.condition_col]
            cond_cats = list(pd.Categorical(cond_series).categories)
            if len(cond_cats) == 0:
                raise ValueError('condition_col has no valid categories after preprocessing')

            cond_values = pd.Categorical(cond_series, categories=cond_cats).codes.astype(int)
            valid_cond_mask = cond_values >= 0
            if not np.any(valid_cond_mask):
                raise ValueError('condition_col contains only missing values after preprocessing')

            if not np.all(valid_cond_mask):
                fallback_class = int(cond_values[valid_cond_mask][0])
                cond_values = cond_values.copy()
                cond_values[~valid_cond_mask] = fallback_class

            # always remove condition from feature matrix to avoid leakage/duplication
            feature_df = df.drop(columns=[self.condition_col])

        feature_columns = set(feature_df.columns)
        feature_continuous_cols = [c for c in self.continuous_cols if c in feature_columns and c != self.condition_col]
        feature_categorical_cols = [c for c in self.categorical_cols if c in feature_columns and c != self.condition_col]
        feature_binary_cols = [c for c in self.binary_cols if c in feature_columns and c != self.condition_col]
        feature_ordinal_cols = [c for c in self.ordinal_cols if c in feature_columns and c != self.condition_col]
        feature_ordinal_orders = {
            col: order
            for col, order in self.ordinal_orders.items()
            if col in feature_ordinal_cols
        }

        self.preprocessor = SimplePreprocessor(
            feature_continuous_cols,
            feature_categorical_cols,
            feature_binary_cols,
            feature_ordinal_cols,
            feature_ordinal_orders,
        )
        self.preprocessor.fit(feature_df)
        X = self.preprocessor.transform(feature_df)
        if cond_values is not None:
            # create one-hot condition vector
            cond_oh = np.zeros((len(df), len(cond_cats)), dtype=np.float32)
            for i, v in enumerate(cond_values):
                cond_oh[i, v] = 1.0
        else:
            cond_oh = None

        dataset = TabularDataset(X, cond_oh)
        drop_last_batches = len(dataset) >= self.batch_size
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=drop_last_batches)

        input_dim = self.preprocessor.output_dim
        cond_dim = cond_oh.shape[1] if cond_oh is not None else 0
        aux_classes = None
        if self.aux_info_loss and cond_values is not None:
            aux_classes = int(cond_dim)

        # build models
        self._build(input_dim=input_dim, cond_dim=cond_dim, aux_classes=aux_classes)

        safe_progress_call({
            'event': 'training_started',
            'epoch': 0,
            'total_epochs': self.epochs,
        })

        # training loop with early stopping
        iters = 0
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            batch_count = 0
            for batch in loader:
                iters += 1
                if cond_oh is not None:
                    real, cond = batch
                    cond = cond.to(self.device)
                else:
                    real = batch
                    cond = None
                real = real.to(self.device)

                # pack pac times to reduce mode collapse (as in CTGAN)
                if self.pac > 1:
                    real_packed = real
                    cond_packed = cond
                else:
                    real_packed = real
                    cond_packed = cond

                # -----------------
                # train discriminator
                # -----------------
                for _ in range(self.n_critic):
                    z = torch.randn(real_packed.shape[0], self.noise_dim, device=self.device)
                    if cond_packed is not None:
                        # optionally tile condition to noise vector (or concatenate if sizes differ)
                        c_for_G = cond_packed
                    else:
                        c_for_G = None
                    fake = self.G(z, c_for_G)

                    self.optim_D.zero_grad()

                    if self.aux_info_loss:
                        real_out = self.D(real_packed, cond_packed)
                        if isinstance(real_out, tuple):
                            real_rf, real_aux = real_out
                        else:
                            real_rf = real_out
                            real_aux = None

                        fake_out = self.D(fake.detach(), cond_packed)
                        if isinstance(fake_out, tuple):
                            fake_rf, fake_aux = fake_out
                        else:
                            fake_rf = fake_out
                            fake_aux = None
                    else:
                        real_rf = self.D(real_packed, cond_packed)
                        fake_rf = self.D(fake.detach(), cond_packed)

                    # discriminator loss
                    if self.wgan_gp:
                        d_loss = fake_rf.mean() - real_rf.mean()
                        if self.gp_weight > 0:
                            gp = gradient_penalty(self.D, real_packed, fake.detach(), device=self.device, c=cond_packed)
                            d_loss = d_loss + self.gp_weight * gp
                    elif self.hinge_loss:
                        # hinge loss from SAGAN
                        d_loss = torch.mean(F.relu(1. - real_rf)) + torch.mean(F.relu(1. + fake_rf))
                    else:
                        # vanilla BCE
                        real_labels = torch.ones_like(real_rf)
                        fake_labels = torch.zeros_like(fake_rf)
                        d_loss = F.binary_cross_entropy_with_logits(real_rf, real_labels) + \
                                 F.binary_cross_entropy_with_logits(fake_rf, fake_labels)

                    # auxiliary info loss (for discriminator)
                    if self.aux_info_loss and real_aux is not None:
                        # cond_packed contains one-hot vectors; convert to class indices
                        # we assume cond_packed was derived from same cond_values
                        # for packed mode cond_packed is one-hot blocks - simpler path: use non-packed aux if pac==1
                        if self.pac == 1:
                            # get indices
                            cond_idx = torch.argmax(cond_packed, dim=1).long()
                            aux_loss_real = self.aux_criterion(real_aux, cond_idx.to(self.device))
                            d_loss = d_loss + self.aux_loss_weight * aux_loss_real

                    d_loss.backward()
                    self.optim_D.step()

                # -----------------
                # train generator
                # -----------------
                z = torch.randn(real_packed.shape[0], self.noise_dim, device=self.device)
                if cond_packed is not None:
                    c_for_G = cond_packed
                else:
                    c_for_G = None
                fake = self.G(z, c_for_G)
                self.optim_G.zero_grad()

                if self.aux_info_loss:
                    fake_out = self.D(fake, cond_packed)
                    if isinstance(fake_out, tuple):
                        fake_rf, fake_aux = fake_out
                    else:
                        fake_rf = fake_out
                        fake_aux = None
                else:
                    fake_rf = self.D(fake, cond_packed)

                if self.wgan_gp:
                    g_loss = -fake_rf.mean()
                elif self.hinge_loss:
                    g_loss = -torch.mean(fake_rf)
                else:
                    labels = torch.ones_like(fake_rf)
                    g_loss = F.binary_cross_entropy_with_logits(fake_rf, labels)

                # generator auxiliary loss to encourage mutual information
                if self.aux_info_loss and fake_aux is not None and self.pac == 1:
                    cond_idx = torch.argmax(cond_packed, dim=1).long()
                    aux_loss_gen = self.aux_criterion(fake_aux, cond_idx.to(self.device))
                    g_loss = g_loss + self.aux_loss_weight * aux_loss_gen

                if self.correlation_preservation and self.correlation_loss_weight > 0:
                    corr_loss = self._correlation_preservation_loss(real_packed.detach(), fake)
                    g_loss = g_loss + self.correlation_loss_weight * corr_loss

                g_loss.backward()
                self.optim_G.step()

                # track losses for visualization
                self.d_losses.append(float(d_loss))
                self.g_losses.append(float(g_loss))
                epoch_d_loss += float(d_loss)
                epoch_g_loss += float(g_loss)
                batch_count += 1

                # logging per iteration
                if self.verbose and iters % 100 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}] Iter {iters} d_loss={float(d_loss):.4f} g_loss={float(g_loss):.4f}")
            
            # end of epoch - compute average losses and check for early stopping
            avg_epoch_g_loss = epoch_g_loss / batch_count if batch_count > 0 else 0
            avg_epoch_d_loss = epoch_d_loss / batch_count if batch_count > 0 else 0
            avg_epoch_loss = (avg_epoch_g_loss + avg_epoch_d_loss) / 2

            safe_progress_call({
                'event': 'epoch_end',
                'epoch': epoch + 1,
                'total_epochs': self.epochs,
                'avg_g_loss': float(avg_epoch_g_loss),
                'avg_d_loss': float(avg_epoch_d_loss),
            })
            
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs}: Avg G Loss={avg_epoch_g_loss:.6f}, Avg D Loss={avg_epoch_d_loss:.6f}")
            
            # early stopping check (only after 50 epochs minimum)
            if self.early_stopping and epoch >= 49:  # epoch is 0-indexed, so epoch 49 = 50th epoch
                if avg_epoch_loss < best_loss - self.min_delta:
                    # improvement found
                    best_loss = avg_epoch_loss
                    patience_counter = 0
                    if self.verbose:
                        print(f"  [+] Loss improved to {best_loss:.6f}")
                else:
                    # no improvement
                    patience_counter += 1
                    if self.verbose:
                        print(f"  [!] No improvement ({patience_counter}/{self.patience})")
                    if patience_counter >= self.patience:
                        print(f"\n>> Early stopping: No improvement for {self.patience} epochs after 50 epochs. Stopping at epoch {epoch+1}/{self.epochs}")
                        break

        safe_progress_call({
            'event': 'training_finished',
            'epoch': min(self.epochs, len(self.g_losses)),
            'total_epochs': self.epochs,
        })

        # end training
        return self

    def sample(self, n: int, condition: Optional[int] = None) -> pd.DataFrame:
        """Generate n synthetic rows. If condition is provided (index of category), returns samples with that condition.
        For condition passing: if the model was trained with a condition, pass the index of the category.
        """
        if self.preprocessor is None or self.G is None:
            raise RuntimeError('Model not trained yet. Call fit() first.')

        # use the stored condition vector length (set in _build) to construct condition tensors.
        # relying on the preprocessor for condition size is fragile because condition was
        # handled separately during training; _cond_dim reflects what the generator expects.
        cond_dim = getattr(self, '_cond_dim', 0)

        out = []
        batch = 512
        while len(out) < n:
            b = min(batch, n - len(out))
            z = torch.randn(b, self.noise_dim, device=self.device)
            if condition is None and cond_dim and cond_dim > 0:
                # sample random condition indices
                idxs = np.random.randint(0, cond_dim, size=b)
                cond_batch = np.zeros((b, cond_dim), dtype=np.float32)
                for i, idx in enumerate(idxs):
                    cond_batch[i, idx] = 1.0
                cond_tensor = torch.from_numpy(cond_batch).to(self.device)
                fake = self.G(z, cond_tensor)
            elif condition is not None and cond_dim and cond_dim > 0:
                cond_batch = np.zeros((b, cond_dim), dtype=np.float32)
                for i in range(b):
                    cond_batch[i, condition] = 1.0
                cond_tensor = torch.from_numpy(cond_batch).to(self.device)
                fake = self.G(z, cond_tensor)
            else:
                fake = self.G(z, None)
            fake = fake.detach().cpu().numpy()
            out.append(fake)
        out = np.vstack(out)[:n]
        df = self.preprocessor.inverse_transform(out)
        return df

    def save(self, path: str):
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        # save model weights, preprocessor and config metadata
        state = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'preprocessor': {
                'continuous_cols': self.preprocessor.continuous_cols,
                'categorical_cols': self.preprocessor.categorical_cols,
                'binary_cols': self.preprocessor.binary_cols,
                'ordinal_cols': self.preprocessor.ordinal_cols,
                'ordinal_categories': self.preprocessor.ordinal_categories,
                'cat_categories': self.preprocessor.cat_categories,
                'col_order': self.preprocessor.col_order,
                'col_slices': {k: (v.start, v.stop) for k,v in self.preprocessor.col_slices.items()},
                '_vector_dim': self.preprocessor._vector_dim
            },
            'config': {
                'noise_dim': self.noise_dim,
                'generator_dim': tuple(self.generator_dim),
                'discriminator_dim': tuple(self.discriminator_dim),
                'embedding_dim': self.embedding_dim,
                'cond_dim': getattr(self, '_cond_dim', 0),
                'condition_col': self.condition_col,
                'aux_info_loss': self.aux_info_loss,
                'aux_loss_weight': self.aux_loss_weight,
                'pac': self.pac
            }
        }

        torch.save(state, path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        # reconstruct preprocessor
        p = SimplePreprocessor([], [])
        pp = state['preprocessor']
        p.continuous_cols = pp['continuous_cols']
        p.categorical_cols = pp['categorical_cols']
        p.binary_cols = pp['binary_cols']
        p.ordinal_cols = pp.get('ordinal_cols', [])
        p.ordinal_categories = pp.get('ordinal_categories', {})
        p.ordinal_orders = {k: list(v) for k, v in p.ordinal_categories.items()}
        p.cat_categories = pp['cat_categories']
        p.col_order = pp['col_order']
        p.col_slices = {k: slice(v[0], v[1]) for k,v in pp['col_slices'].items()}
        p._vector_dim = pp['_vector_dim']
        self.preprocessor = p

        # build architectures
        input_dim = p._vector_dim
        # prefer config metadata saved in checkpoint; fallback to preprocessor slice if missing
        cfg = state.get('config', {})
        cond_dim = cfg.get('cond_dim', 0)
        if cond_dim == 0 and self.condition_col and self.condition_col in p.col_slices:
            sl = p.col_slices[self.condition_col]
            cond_dim = sl.stop - sl.start

        aux_classes = None
        if self.aux_info_loss:
            aux_classes = None
            # cannot recover aux classes easily; user responsible to set same config

        self._build(input_dim=input_dim, cond_dim=cond_dim, aux_classes=aux_classes)
        self.G.load_state_dict(state['G'])
        self.D.load_state_dict(state['D'])
        # restore some config fields if present
        if 'config' in state:
            conf = state['config']
            # update a subset of runtime params so sampling and further saves are consistent
            self.noise_dim = conf.get('noise_dim', self.noise_dim)
            self.generator_dim = tuple(conf.get('generator_dim', self.generator_dim))
            self.discriminator_dim = tuple(conf.get('discriminator_dim', self.discriminator_dim))
            self.embedding_dim = conf.get('embedding_dim', self.embedding_dim)
            # ensure we remember cond dim set at build time
            self._cond_dim = conf.get('cond_dim', getattr(self, '_cond_dim', 0))


class CorrelationAwareCTGAN(CustomCTGAN):
    def __init__(self, correlation_loss_weight: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.correlation_preservation = True
        self.correlation_loss_weight = float(correlation_loss_weight)


class OrdinalCTGAN(CustomCTGAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# ----------------------------- Example Usage ----------------------------

if __name__ == '__main__':
    # small example to demonstrate usage
    # > create a tiny synthetic dataset
    n = 2000
    df = pd.DataFrame({
        'Age': np.random.normal(60, 10, size=n).clip(18, 90),
        'Years_treatment': np.random.exponential(2, size=n),
        'Sex': np.random.choice(['M', 'F'], size=n, p=[0.48, 0.52]),
        'Ethnicity': np.random.choice(['A','B','C'], size=n, p=[0.7,0.2,0.1]),
        'Location': np.random.choice(['X','Y','Z'], size=n),
        'Outcome': np.random.choice(['Good','Bad'], size=n, p=[0.8,0.2]),
        'Treat': np.random.choice(['A','B'], size=n, p=[0.6,0.4])
    })

    continuous = ['Age', 'Years_treatment']
    categorical = ['Sex', 'Ethnicity', 'Location', 'Outcome', 'Treat']

    model = CustomCTGAN(
        continuous_cols=continuous,
        categorical_cols=['Ethnicity', 'Location'],
        binary_cols=['Sex', 'Outcome'],
        condition_col='Treat',
        noise_dim=64,
        generator_dim=(256,256),
        discriminator_dim=(256,256),
        embedding_dim=64,
        batch_size=256,
        n_critic=5,
        pac=1,
        lr_g=2e-4,
        lr_d=2e-4,
        epochs=5,
        wgan_gp=True,
        gp_weight=10.0,
        hinge_loss=False,
        dropout=0.0,
        spectral_norm=False,
        aux_info_loss=False,
        verbose=True
    )

    model.fit(df)
    synth = model.sample(10, condition=0)
    print(synth.head())
    model.save('ctgan_custom_checkpoint.pth')

