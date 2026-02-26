import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from copy import deepcopy

# =============================
# GPU Setup
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# =============================
# 1. Simulate Data
# =============================

np.random.seed(42)
torch.manual_seed(42)

N = 10000
T_1 = 6
T = 5
T_input = 4

mu = 100
tau2 = 400
sigma2 = 900

tau = np.sqrt(tau2)
sigma = np.sqrt(sigma2)

theta = np.random.normal(mu, tau, N)

X = np.array([np.random.normal(theta[i], sigma, T_1) for i in range(N)])


# =============================
# 1b. NORMALIZE DATA
# =============================
# Normalization helps networks learn better, especially for extreme values
X_mean = X.mean()
X_std = X.std()
X_normalized = (X - X_mean) / X_std
theta_normalized = (theta - X_mean) / X_std

X_input = X_normalized[:, :T_input]
Y_target = X_normalized[:, T_input]

# =============================
# 2. Classical Bühlmann
# =============================

K = sigma2 / tau2
Z = T_input / (T_input + K)

X_bar = X_input.mean(axis=1)
buhlmann_pred = Z * X_bar + (1 - Z) * 0

# Denormalize Bühlmann predictions
buhlmann_pred = buhlmann_pred * X_std + X_mean
# Denormalize Y_target for later comparison
Y_target_original = Y_target * X_std + X_mean

# =============================
# 3. Train / Validation Split
# =============================

idx = np.random.permutation(N)
train_size = int(0.8 * N)

train_idx = idx[:train_size]
val_idx = idx[train_size:]

X_train = torch.tensor(X_input[train_idx], dtype=torch.float32)
Y_train = torch.tensor(Y_target[train_idx], dtype=torch.float32).unsqueeze(1)

X_val = torch.tensor(X_input[val_idx], dtype=torch.float32)
Y_val = torch.tensor(Y_target[val_idx], dtype=torch.float32).unsqueeze(1)

train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)

# =============================
# 4. Transformer Model
# =============================


class CredibilityTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Linear(1, 32)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=64, batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.head = nn.Linear(32, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x)


class CredibilityTransformerMasked(nn.Module):
    """Transformer that accepts a 2-channel input: (value, mask).

    The mask is a binary indicator (1 = real, 0 = padded). The model
    learns to ignore padded positions using the mask as an input feature.
    """

    def __init__(self):
        super().__init__()

        self.embedding = nn.Linear(2, 32)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=64, batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.head = nn.Linear(32, 1)

    def forward(self, x):
        # x: (batch, seq_len, 2)
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x)


model = CredibilityTransformer().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# =============================
# 5. Training with Early Stopping
# =============================

epochs = 200
patience = 15
best_val_loss = float("inf")
patience_counter = 0
best_model_state = None

for epoch in range(epochs):

    # ---- Training ----
    model.train()
    train_loss = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---- Validation ----
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
    )

    # ---- Early Stopping ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = deepcopy(model.state_dict())
        best_epoch = epoch
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# Restore best model
model.load_state_dict(best_model_state)

# =============================
# 6. Evaluation
# =============================

model.eval()
with torch.no_grad():
    transformer_pred = (
        model(torch.tensor(X_input, dtype=torch.float32).to(device))
        .cpu()
        .numpy()
        .flatten()
    )
transformer_pred = transformer_pred * X_std + X_mean
mse_transformer = np.mean((transformer_pred - Y_target_original) ** 2)
mse_buhlmann = np.mean((buhlmann_pred - Y_target_original) ** 2)

print("\n===== Final Prediction MSE =====")
print("Transformer MSE:", mse_transformer)
print("Bühlmann MSE:", mse_buhlmann)

r2_between = np.corrcoef(transformer_pred, buhlmann_pred)[0, 1] ** 2
print("R² between Transformer and Bühlmann:", r2_between)

# =============================
# 7. Optional: Retrain on Full Data
# =============================

print("\nRetraining on full dataset...")

# Keep the original 1-4 inputs for evaluation (predicting year 5)
orig_full_X_vals = torch.tensor(X_input, dtype=torch.float32)

# Build moving-interval training pairs: for each policy i, use
# 1 -> 2, 1-2 -> 3, 1-3 -> 4, 1-4 -> 5. Pad shorter inputs to length T_input with -1
# and include a binary mask (1=real, 0=padded) as the second channel.
values_list = []
masks_list = []
targets_list = []
for i in range(N):
    for L in range(1, T_input + 1):  # L = 1..T_input
        inp = X_normalized[i, :L]  # Use normalized data
        if L < T_input:
            pad_vals = np.full(T_input - L, -1.0)
            padded_vals = np.concatenate([inp, pad_vals])
            mask = np.concatenate(
                [np.ones(L, dtype=float), np.zeros(T_input - L, dtype=float)]
            )
        else:
            padded_vals = inp.copy()
            mask = np.ones(T_input, dtype=float)
        target = X_normalized[i, L]  # next year after the L observations (normalized)
        values_list.append(padded_vals)
        masks_list.append(mask)
        targets_list.append(target)

vals_arr = np.array(values_list)
masks_arr = np.array(masks_list)
full_X_aug = torch.tensor(np.stack([vals_arr, masks_arr], axis=-1), dtype=torch.float32)
full_Y_aug = torch.tensor(np.array(targets_list), dtype=torch.float32).unsqueeze(1)

full_dataset = torch.utils.data.TensorDataset(full_X_aug, full_Y_aug)
full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=256, shuffle=True)

# Split augmented full data into training and validation for early stopping
n_aug = full_X_aug.shape[0]
perm = torch.randperm(n_aug)
split = int(0.8 * n_aug)
train_idx = perm[:split]
val_idx = perm[split:]

train_X_aug = full_X_aug[train_idx]
train_Y_aug = full_Y_aug[train_idx]
val_X_aug = full_X_aug[val_idx]
val_Y_aug = full_Y_aug[val_idx]

train_full_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_X_aug, train_Y_aug),
    batch_size=256,
    shuffle=True,
)
val_full_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(val_X_aug, val_Y_aug), batch_size=256, shuffle=False
)

# Train with early stopping on the augmented dataset to find best epoch
model_full = CredibilityTransformerMasked().to(device)
optimizer_full = optim.Adam(model_full.parameters(), lr=0.0001)
epochs_full = 200
patience_full = 15
best_val_loss_full = float("inf")
patience_counter_full = 0
best_model_state_full = None
best_epoch_full = 0

for epoch in range(epochs_full):
    model_full.train()
    train_loss = 0
    for batch_x, batch_y in train_full_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer_full.zero_grad()
        preds = model_full(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer_full.step()
        train_loss += loss.item()
    train_loss /= len(train_full_loader)

    model_full.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_full_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model_full(batch_x)
            loss = criterion(preds, batch_y)
            val_loss += loss.item()
    val_loss /= len(val_full_loader)

    print(
        f"Full Aug Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss_full:
        best_val_loss_full = val_loss
        best_model_state_full = deepcopy(model_full.state_dict())
        best_epoch_full = epoch
        patience_counter_full = 0
    else:
        patience_counter_full += 1
        if patience_counter_full >= patience_full:
            print("Early stopping for full-augmented training triggered.")
            break

# Record best epoch (0-based). Now retrain on the entire augmented dataset for that many epochs.
best_epochs_to_train = best_epoch_full + 1
print(
    f"Best epoch on validation (augmented): {best_epoch_full} -> will retrain full for {best_epochs_to_train} epochs"
)

# Retrain a fresh model on the whole augmented dataset for the chosen number of epochs
model_full_final = CredibilityTransformerMasked().to(device)
optimizer_full_final = optim.Adam(model_full_final.parameters(), lr=0.0001)
full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=256, shuffle=True)
for epoch in range(best_epochs_to_train):
    model_full_final.train()
    total_loss = 0
    for batch_x, batch_y in full_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer_full_final.zero_grad()
        preds = model_full_final(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer_full_final.step()
        total_loss += loss.item()
    print(f"Retrain Full Epoch {epoch+1}, Loss: {total_loss/len(full_loader):.4f}")

# Use the retrained final model for downstream evaluation
model_full = model_full_final
model_full.eval()
with torch.no_grad():
    # create mask=1 for the original full inputs (no padding)
    orig_full_X_vals_gpu = orig_full_X_vals.to(device)
    orig_mask = torch.ones_like(orig_full_X_vals_gpu)
    orig_full_X = torch.stack([orig_full_X_vals_gpu, orig_mask], dim=-1)
    transformer_pred_full_norm = model_full(orig_full_X).cpu().numpy().flatten()
    # Denormalize predictions back to original scale
    transformer_pred_full = transformer_pred_full_norm * X_std + X_mean
# =============================
# 8. Plot Comparison
# =============================
mse_transformer = np.mean((transformer_pred_full - Y_target_original) ** 2)
mse_buhlmann = np.mean((buhlmann_pred - Y_target_original) ** 2)

print("\n===== Final Prediction MSE =====")
print("Transformer MSE:", mse_transformer)
print("Bühlmann MSE:", mse_buhlmann)

r2_between = np.corrcoef(transformer_pred_full, buhlmann_pred)[0, 1] ** 2
print("R² between Transformer and Bühlmann:", r2_between)

# plt.figure()
# plt.scatter(buhlmann_pred[:1000], transformer_pred_full[:1000])
# plt.xlabel("Bühlmann Prediction")
# plt.ylabel("Transformer Prediction")
# plt.title("Transformer vs Bühlmann")
# plt.show()


X_input_1 = X_normalized[:, : (T_input + 1)]
Y_target_1 = X_normalized[:, (T_input + 1)]
Y_target_1_original = Y_target_1 * X_std + X_mean  # Denormalized for comparison

full_X_1_vals = torch.tensor(X_input_1, dtype=torch.float32)
model_full.eval()
with torch.no_grad():
    full_X_1_vals_gpu = full_X_1_vals.to(device)
    mask1 = torch.ones_like(full_X_1_vals_gpu)
    full_X_1_in = torch.stack([full_X_1_vals_gpu, mask1], dim=-1)
    transformer_pred_full_1_norm = model_full(full_X_1_in).cpu().numpy().flatten()
    # Denormalize predictions
    transformer_pred_full_1 = transformer_pred_full_1_norm * X_std + X_mean

mse_transformer = np.mean((transformer_pred_full_1 - Y_target_1_original) ** 2)
mse_buhlmann = np.mean((buhlmann_pred - Y_target_1_original) ** 2)

print("\n===== Final Prediction MSE =====")
print("Transformer MSE:", mse_transformer)
print("Bühlmann MSE:", mse_buhlmann)

r2_between = np.corrcoef(transformer_pred_full_1, buhlmann_pred)[0, 1] ** 2
print("R² between Transformer and Bühlmann:", r2_between)

# plt.figure()
# plt.scatter(buhlmann_pred[:1000], transformer_pred_full_1[:1000])
# plt.xlabel("Bühlmann Prediction")
# plt.ylabel("Transformer Prediction")
# plt.title("Transformer vs Bühlmann")
# plt.show()
# =============================
# Estimate EPV and VHM
# =============================
N = X_input_1.shape[0]
n = X_input_1.shape[1]


# Per-risk mean
X_bar_i = X_input_1.mean(axis=1)

# Overall mean
X_bar = X_bar_i.mean()

# ---- EPV (within variance) ----
EPV_hat = np.sum((X_input_1 - X_bar_i[:, None]) ** 2) / (N * (n - 1))

# ---- Between variance ----
S2_between = np.sum((X_bar_i - X_bar) ** 2) / (N - 1)

# ---- VHM ----
VHM_hat = S2_between - EPV_hat / n

# Safety: VHM cannot be negative
VHM_hat = max(VHM_hat, 1e-8)

# ---- Credibility factor ----
K_hat = EPV_hat / VHM_hat
Z_hat = n / (n + K_hat)

# ---- Bühlmann prediction ----
buhlmann_estimated = Z_hat * X_bar_i + (1 - Z_hat) * X_bar

print("Estimated EPV:", EPV_hat)
print("Estimated VHM:", VHM_hat)
print("Estimated Z:", Z_hat)
# Denormalize Bühlmann predictions for comparison with theta
buhlmann_estimated_original = buhlmann_estimated * X_std + X_mean

mse_transformer = np.mean((transformer_pred_full_1 - theta) ** 2)
mse_buhlmann = np.mean((buhlmann_estimated_original - theta) ** 2)

print("\n===== Final Prediction MSE =====")
print("Transformer MSE:", mse_transformer)
print("Bühlmann MSE:", mse_buhlmann)

r2_between = (
    np.corrcoef(transformer_pred_full_1, buhlmann_estimated_original)[0, 1] ** 2
)
print("R² between Transformer and Bühlmann:", r2_between)

plt.figure()
plt.scatter(buhlmann_estimated_original[:10000], transformer_pred_full_1[:10000], s=9)
min_val = min(
    np.min(buhlmann_estimated_original[:10000]), np.min(transformer_pred_full_1[:10000])
)
max_val = max(
    np.max(buhlmann_estimated_original[:10000]), np.max(transformer_pred_full_1[:10000])
)
plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    color="gray",
    label="45-degree line",
)
plt.xlabel("Bühlmann Prediction")
plt.ylabel("Transformer Prediction")
plt.title("Bühlmann Credibility vs Transformer")
plt.legend()
plt.savefig("buhlmann_credibility_vs_transformer.png")
plt.show(block=False)
print("!")

# =============================
# New Block: Training/Validation with X_modified
# =============================
# For each row in X_modified, use all but the last non-nan as input, last non-nan as target.
# Only include rows with at least 2 non-nan values (so at least 1 for input, 1 for target).


# Create X_modified: for every 6 rows, keep only 1,2,3,4,5,6 years of data respectively, rest set to np.nan
X_modified = X_normalized.copy()  # Use normalized data
for i in range(0, N, 6):
    for j in range(6):
        row = i + j
        if row >= N:
            break
        keep_years = j + 1
        if keep_years < T_1:
            X_modified[row, keep_years:] = np.nan

inputs_mod = []
masks_mod = []
targets_mod = []
for row in X_modified:
    not_nan = np.where(~np.isnan(row))[0]
    if len(not_nan) < 2:
        continue  # skip rows with only 1 year of data
    last_idx = not_nan[-1]
    input_vals = row[:last_idx]
    mask = np.ones_like(input_vals, dtype=float)
    # Pad to length 5 (T_input) with -1 and mask 0
    pad_len = 5 - len(input_vals)
    if pad_len > 0:
        input_vals = np.concatenate([input_vals, np.full(pad_len, -1.0)])
        mask = np.concatenate([mask, np.zeros(pad_len)])
    targets_mod.append(row[last_idx])  # Already normalized
    inputs_mod.append(input_vals)
    masks_mod.append(mask)

inputs_mod = np.array(inputs_mod)
masks_mod = np.array(masks_mod)
targets_mod = np.array(targets_mod)

# Split into train/val
num_mod = len(inputs_mod)
perm_mod = np.random.permutation(num_mod)
split_mod = int(0.8 * num_mod)
train_idx_mod = perm_mod[:split_mod]
val_idx_mod = perm_mod[split_mod:]

X_train_mod = torch.tensor(
    np.stack([inputs_mod[train_idx_mod], masks_mod[train_idx_mod]], axis=-1),
    dtype=torch.float32,
)
Y_train_mod = torch.tensor(targets_mod[train_idx_mod], dtype=torch.float32).unsqueeze(1)
X_val_mod = torch.tensor(
    np.stack([inputs_mod[val_idx_mod], masks_mod[val_idx_mod]], axis=-1),
    dtype=torch.float32,
)
Y_val_mod = torch.tensor(targets_mod[val_idx_mod], dtype=torch.float32).unsqueeze(1)

train_dataset_mod = torch.utils.data.TensorDataset(X_train_mod, Y_train_mod)
val_dataset_mod = torch.utils.data.TensorDataset(X_val_mod, Y_val_mod)
train_loader_mod = torch.utils.data.DataLoader(
    train_dataset_mod, batch_size=256, shuffle=True
)
val_loader_mod = torch.utils.data.DataLoader(
    val_dataset_mod, batch_size=256, shuffle=False
)

# Train masked transformer with early stopping
model_modified = CredibilityTransformerMasked()
optimizer_mod = optim.Adam(model_modified.parameters(), lr=0.0001)
epochs_mod = 200
patience_mod = 15
best_val_loss_mod = float("inf")
patience_counter_mod = 0
best_model_state_mod = None
best_epoch_mod = 0
for epoch in range(epochs_mod):
    model_modified.train()
    train_loss = 0
    for batch_x, batch_y in train_loader_mod:
        optimizer_mod.zero_grad()
        preds = model_modified(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer_mod.step()
        train_loss += loss.item()
    train_loss /= len(train_loader_mod)

    model_modified.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader_mod:
            preds = model_modified(batch_x)
            loss = criterion(preds, batch_y)
            val_loss += loss.item()
    val_loss /= len(val_loader_mod)

    print(
        f"X_mod Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss_mod:
        best_val_loss_mod = val_loss
        best_model_state_mod = deepcopy(model_modified.state_dict())
        best_epoch_mod = epoch
        patience_counter_mod = 0
    else:
        patience_counter_mod += 1
        if patience_counter_mod >= patience_mod:
            print("Early stopping for X_modified training triggered.")
            break

# Retrain on all data for best number of epochs
best_epochs_mod = best_epoch_mod + 1
print(
    f"Best epoch for X_modified: {best_epoch_mod} -> retrain for {best_epochs_mod} epochs"
)
model_modified_full = CredibilityTransformerMasked()
optimizer_mod_full = optim.Adam(model_modified_full.parameters(), lr=0.0001)
full_dataset_mod = torch.utils.data.TensorDataset(
    torch.tensor(np.stack([inputs_mod, masks_mod], axis=-1), dtype=torch.float32),
    torch.tensor(targets_mod, dtype=torch.float32).unsqueeze(1),
)
full_loader_mod = torch.utils.data.DataLoader(
    full_dataset_mod, batch_size=256, shuffle=True
)
for epoch in range(best_epochs_mod):
    model_modified_full.train()
    total_loss = 0
    for batch_x, batch_y in full_loader_mod:
        optimizer_mod_full.zero_grad()
        preds = model_modified_full(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer_mod_full.step()
        total_loss += loss.item()
    print(
        f"Retrain X_mod Full Epoch {epoch+1}, Loss: {total_loss/len(full_loader_mod):.4f}"
    )

# Keep the original 1-4 inputs for evaluation (predicting year 5)
orig_full_X_modified_vals = torch.tensor(X_modified, dtype=torch.float32)

# Build moving-interval training pairs for X_modified: for each policy i, use
# 1 -> 2, 1-2 -> 3, 1-3 -> 4, 1-4 -> 5. Pad shorter inputs to length T_input with -1
# and include a binary mask (1=real, 0=padded) as the second channel.
inputs_mod = []
masks_mod = []
# targets_mod = []
for row in X_modified:
    not_nan = np.where(~np.isnan(row))[0]
    last_idx = not_nan[-1]
    input_vals = row[: last_idx + 1]
    mask = np.ones_like(input_vals, dtype=float)
    # Pad to length 5 (T_input) with -1 and mask 0
    pad_len = T_1 - len(input_vals)
    if pad_len > 0:
        input_vals = np.concatenate([input_vals, np.full(pad_len, -1.0)])
        mask = np.concatenate([mask, np.zeros(pad_len)])
    # targets_mod.append(row[last_idx])
    inputs_mod.append(input_vals)
    masks_mod.append(mask)

inputs_mod = np.array(inputs_mod)
masks_mod = np.array(masks_mod)
# targets_mod = np.array(targets_mod)

model_modified_full.eval()
with torch.no_grad():
    # Prepare X_modified for evaluation: replace NaN with -1 and create mask
    orig_full_X_modified_vals_filled = torch.tensor(inputs_mod, dtype=torch.float32)
    orig_mask = torch.tensor(masks_mod, dtype=torch.float32)
    orig_full_X_modified = torch.stack(
        [orig_full_X_modified_vals_filled, orig_mask], dim=-1
    )
    transformer_pred_full_norm = (
        model_modified_full(orig_full_X_modified).numpy().flatten()
    )
    # Denormalize predictions back to original scale
    transformer_pred_full = transformer_pred_full_norm * X_std + X_mean

print("!")
# =============================
# Estimate EPV and VHM
# =============================
N = X_modified.shape[0]

# Bühlmann credibility weighting for X_modified (each row = each class, NaN = missing years)
n = np.sum(~np.isnan(X_modified), axis=1)  # number of years per class
X_bar_i = np.nanmean(X_modified, axis=1)  # per-class mean
X_bar = np.nanmean(X_modified)  # overall mean

# EPV: within-class variance (average over all available years)
EPV_hat = np.nansum((X_modified - X_bar_i[:, None]) ** 2) / (np.sum(n - 1))

# Between-class variance
S2_between = np.nansum((X_bar_i - X_bar) ** 2) / (N - 1)

# VHM: variance of hypothetical means

# Correct VHM_hat calculation: VHM_hat = S2_between - EPV_hat * mean(1/n)
mean_inv_n = np.mean(1 / n)
VHM_hat = S2_between - EPV_hat * mean_inv_n
VHM_hat = max(VHM_hat, 1e-8)

# Credibility factor
K_hat = EPV_hat / VHM_hat
Z_hat = n / (n + K_hat)

# Bühlmann prediction (in normalized space)
buhlmann_estimated = Z_hat * X_bar_i + (1 - Z_hat) * X_bar
# Denormalize for comparison
buhlmann_estimated_original = buhlmann_estimated * X_std + X_mean

# print("Estimated EPV:", EPV_hat)
# print("Estimated VHM:", VHM_hat)
# print("Estimated Z:", Z_hat)
mse_transformer = np.mean((transformer_pred_full - theta) ** 2)
mse_buhlmann = np.mean((buhlmann_estimated_original - theta) ** 2)

print("\n===== Final Prediction MSE =====")
print("Transformer MSE:", mse_transformer)
print("Bühlmann MSE:", mse_buhlmann)

r2_between = np.corrcoef(transformer_pred_full, buhlmann_estimated_original)[0, 1] ** 2
print("R² between Transformer and Bühlmann:", r2_between)

plt.figure()
plt.scatter(buhlmann_estimated_original[:10000], transformer_pred_full[:10000], s=5)
min_val = min(
    np.min(buhlmann_estimated_original[:10000]), np.min(transformer_pred_full[:10000])
)
max_val = max(
    np.max(buhlmann_estimated_original[:10000]), np.max(transformer_pred_full[:10000])
)
plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    color="gray",
    label="45-degree line",
)
plt.xlabel("Bühlmann Prediction")
plt.ylabel("Transformer Prediction")
plt.title("Bühlmann Credibility vs Transformer")
plt.legend()
plt.savefig("buhlmann_credibility_vs_transformer_modifed.png")
plt.show(block=False)

plt.figure()
plt.scatter(buhlmann_estimated_original[:10000], theta[:10000], s=5)
min_val = min(np.min(buhlmann_estimated_original[:10000]), np.min(theta[:10000]))
max_val = max(np.max(buhlmann_estimated_original[:10000]), np.max(theta[:10000]))
plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    color="gray",
    label="45-degree line",
)
plt.xlabel("Bühlmann Prediction")
plt.ylabel("True Mean")
plt.title("Bühlmann Credibility vs True Mean")
plt.legend()
plt.savefig("buhlmann_credibility_vs_true_mean.png")
plt.show(block=False)

plt.figure()
plt.scatter(transformer_pred_full[:10000], theta[:10000], s=5)
min_val = min(np.min(transformer_pred_full[:10000]), np.min(theta[:10000]))
max_val = max(np.max(transformer_pred_full[:10000]), np.max(theta[:10000]))
plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    color="gray",
    label="45-degree line",
)
plt.xlabel("Transformer Prediction")
plt.ylabel("True Mean")
plt.title("Transformer vs True Mean")
plt.legend()
plt.savefig("transformer_vs_true_mean.png")
plt.show(block=False)
