# ───────────────────────── Train (Two-Stage Adaptive) ───────────────────────── #
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

from data  import X, n, p, L, G
from model import theta, MLPwithGamma, get_related_interactions

# ---- config ----
device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_dim     = 50
epochs_stage1  = 1000          # non-adaptive stage
epochs_stage2  = 1000          # adaptive stage
lr             = 1e-3
lambda_theta   = 0.1          # L1 on theta (lasso / adaptive lasso)
lambda_gamma   = 0.1          # group-lasso on first layer columns (per feature)
gamma_value    = 1.0          # power γ in adaptive weights
eps            = 1e-5

# ---- tensors & parameter placement ----
X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
theta.data = theta.data.to(device)
theta.requires_grad_(True)
theta.retain_grad()

# ---- models on device ----
models = [MLPwithGamma(input_dim=p-1).to(device) for _ in range(p)]
assert models[0].gamma.weight.shape == (hidden_dim, p-1)

# ---- upper-tri mapping (1<=i<j<=L → 0-based index) ----
def pair_to_ut_idx0(i, j, L):
    return (i - 1) * L - (i - 1) * i // 2 + (j - i - 1)

# ---- precompute modulation index for each j ----
# For each target column j: build an int64 vector mod_idx[j] of length (p-1),
# where mod_idx[j][t] = -1 if same group as j, else = index in theta for (min(g_j,g_k), max(...))
idx_list = []
mod_idx_list = []
for j in range(p):
    idx = np.array([k for k in range(p) if k != j], dtype=np.int64)
    gj  = G[j]
    mod_idx = np.full(p-1, fill_value=-1, dtype=np.int64)
    for t, k in enumerate(idx):
        gk = G[k]
        if gk != gj:
            a, b = (gj, gk) if gj < gk else (gk, gj)
            mod_idx[t] = pair_to_ut_idx0(a, b, L)
    idx_list.append(torch.tensor(idx, dtype=torch.long, device=device))
    mod_idx_list.append(torch.tensor(mod_idx, dtype=torch.long, device=device))

# ---- proximal operators ----
def soft_threshold(x, thresh):
    return torch.sign(x) * torch.clamp(torch.abs(x) - thresh, min=0.0)

def group_prox(Gw, thresh, weights):
    # Gw: [hidden_dim, p-1], weights: [p-1] nonnegative
    out = Gw.clone()
    for k in range(Gw.shape[1]):
        col = Gw[:, k]
        nrm = col.norm()
        lam = thresh * weights[k]
        factor = torch.clamp(1.0 - lam / (nrm + eps), min=0.0)
        out[:, k] = col * factor
    return out

# ---- helper: one training epoch given weights w_theta, w_gamma_list ----
@torch.no_grad()
def apply_proximal_updates(models, theta, lr, lambda_theta, w_theta, lambda_gamma, w_gamma_list):
    # update non-gamma params by gradient step
    for m in models:
        for name, param in m.named_parameters():
            if name != "gamma.weight":
                param -= lr * param.grad

    # group-prox on gamma weights
    for j, m in enumerate(models):
        G_w = m.gamma.weight
        G_tmp = G_w - lr * G_w.grad
        m.gamma.weight.copy_(group_prox(G_tmp, lambda_gamma * lr, w_gamma_list[j]))

    # soft-threshold on theta
    theta_tmp = theta - lr * theta.grad
    theta.copy_(soft_threshold(theta_tmp, lambda_theta * lr * w_theta))

def one_epoch(models, theta, stage_name, w_theta, w_gamma_list):
    # zero grads
    for m in models:
        m.zero_grad()
    theta.grad = None

    total_mse = 0.0
    for j in range(p):
        m = models[j]
        m.train()
        idx = idx_list[j]                        # (p-1,)
        X_in = X_tensor.index_select(1, idx)     # (n, p-1)
        y_true = X_tensor[:, j]                  # (n,)

        # build modulation vector: ones where same group, theta[...] elsewhere
        mod_idx = mod_idx_list[j]                # (p-1,)
        modulation = torch.ones(p-1, device=device)
        mask = mod_idx >= 0
        if mask.any():
            modulation[mask] = theta.index_select(0, mod_idx[mask])

        y_pred = m(X_in * modulation.unsqueeze(0))
        total_mse += ((y_pred - y_true) ** 2).mean()

    avg_mse = total_mse / p
    avg_mse.backward()
    apply_proximal_updates(models, theta, lr, lambda_theta, w_theta, lambda_gamma, w_gamma_list)

    # clear grads
    for m in models:
        m.zero_grad()
    theta.grad = None
    return avg_mse.item()

# ----------------------------- Stage 1: non-adaptive -----------------------------
w_theta_stage1 = torch.ones_like(theta, device=device)
w_gamma_stage1 = [torch.ones(m.gamma.weight.shape[1], device=device) for m in models]

for epoch in trange(1, epochs_stage1 + 1, desc="Stage 1 (non-adaptive)"):
    mse = one_epoch(models, theta, "stage1", w_theta_stage1, w_gamma_stage1)
    if epoch % 50 == 0 or epoch == 1:
        l1_pen = (theta.abs()).sum().item()
        grp_pen = sum(m.gamma.weight.norm(dim=0).sum().item() for m in models)
        print(f"[S1 {epoch:4d}] MSE={mse:.4f} | L1θ={l1_pen:.4f} | GrpΓ={grp_pen:.4f}")

# ----------------------------- Stage 2: adaptive -----------------------------
# freeze the adaptive weights from stage-1 estimates
with torch.no_grad():
    theta_hat = theta.detach().clone()
    w_theta_stage2 = 1.0 / (theta_hat.abs() ** gamma_value + eps)

    w_gamma_stage2 = []
    for m in models:
        Gh = m.gamma.weight.detach().clone()         # [hidden_dim, p-1]
        # group norm per feature/column
        col_norm = Gh.norm(dim=0)                    # [p-1]
        w_gamma_stage2.append(1.0 / (col_norm ** gamma_value + eps))

for epoch in trange(1, epochs_stage2 + 1, desc="Stage 2 (adaptive)"):
    mse = one_epoch(models, theta, "stage2", w_theta_stage2, w_gamma_stage2)
    if epoch % 50 == 0 or epoch == 1:
        l1_pen = (w_theta_stage2 * theta.abs()).sum().item()
        grp_pen = 0.0
        for j, m in enumerate(models):
            grp_pen += (w_gamma_stage2[j] * m.gamma.weight.norm(dim=0)).sum().item()
        print(f"[S2 {epoch:4d}] MSE={mse:.4f} | L1θ(adap)={l1_pen:.4f} | GrpΓ(adap)={grp_pen:.4f}")

print("Training complete.")
