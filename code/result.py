import numpy as np
import torch

from data  import p, L, G
from model import theta
from train import models      # assumes you run in same project and train.py exposes `models`

# Connection threshold: treat any weight below this as zero
eps_conn = 1e-2

# Move theta to CPU numpy
theta_np = theta.detach().cpu().numpy()

# Extract gamma weights (first-layer) for each of the p models
gamma_weights = [m.gamma.weight.detach().cpu().numpy() for m in models]

# Convert group labels to numpy
G_np = np.array(G)

def theta_index(l, t):
    """
    Return the index in the 1D theta vector corresponding to interaction (l, t), with 1 <= l < t <= L.
    """
    idx = 0
    for i in range(1, L):
        for j in range(i+1, L+1):
            if (i, j) == (l, t):
                return idx
            idx += 1
    raise ValueError(f"No theta index for groups {l},{t}")

def build_adjacency_matrix():
    """
    Build a symmetric p x p adjacency matrix W where:
    - For each pair (j, k):
      * If they share the same group, treat theta_val = 1.
      * Else theta_val = |theta_{lt}| for groups l=G[j], t=G[k].
      * Compute gamma norms norm_jk and norm_kj.
      * Connection exists if theta_val * norm_jk > eps_conn or theta_val * norm_kj > eps_conn.
      * Weight W[j,k] = theta_val * (norm_jk + norm_kj) / 2.
    """
    W = np.zeros((p, p))
    for j in range(p):
        for k in range(j+1, p):
            l, t = G_np[j], G_np[k]
            if l == t:
                theta_val = 1.0
            else:
                idx = theta_index(min(l, t), max(l, t))
                theta_val = abs(theta_np[idx])

            # compute norms of gamma columns
            idxs_j = [x for x in range(p) if x != j]
            pos_k = idxs_j.index(k)
            norm_jk = np.linalg.norm(gamma_weights[j][:, pos_k])

            idxs_k = [x for x in range(p) if x != k]
            pos_j = idxs_k.index(j)
            norm_kj = np.linalg.norm(gamma_weights[k][:, pos_j])

            # determine connection
            conn = (theta_val * norm_jk > eps_conn) or (theta_val * norm_kj > eps_conn)
            if conn:
                W[j, k] = W[k, j] = 1

    return W

if __name__ == "__main__":
    W = build_adjacency_matrix()
    # Save adjacency matrix to CSV
    np.savetxt("adjacency_matrix.csv", W, delimiter=",")
    print("Adjacency matrix saved to 'adjacency_matrix.csv'.")
# -*- coding: utf-8 -*-

