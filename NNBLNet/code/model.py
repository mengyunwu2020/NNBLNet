import torch
import torch.nn as nn
from data import L,n,p
#create theta
theta_dim = L * (L - 1) // 2
theta = torch.nn.Parameter(torch.randn(theta_dim, dtype=torch.float32))


class MLPwithGamma(nn.Module):
    def __init__(self, input_dim, hidden_dim=50):
        super().__init__()
        # First layer weight is considered Gamma
        self.gamma = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.activation(self.gamma(x))      # First layer: Gamma
        out = self.activation(self.layer2(out))   # Second layer
        out = self.layer3(out)                    # Output layer
        return out.squeeze(-1)                    # shape: (batch,)

# Create a list of p neural networks, one for each variable j
mlp_list = [MLPwithGamma(input_dim=p-1) for _ in range(p)]

def get_related_interactions(theta, L, l):
    idx = 0
    related_indices = []
    for i in range(1, L):
        for j in range(i+1, L+1):
            if i == l or j == l:
                related_indices.append(idx)
            idx += 1
    return theta[related_indices]

