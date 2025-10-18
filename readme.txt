#DeepBLNet: Bilevel Network Learning via Hierarchically Structured Sparsity

This repository implements a two-stage proximal gradient descent algorithm with adaptive bi-level sparse penalties for learning a dependency network among p variables.

Requirements:

* Python 3.7 or higher
* NumPy
* Pandas
* PyTorch
* tqdm

Install the required packages with:
pip install numpy pandas torch tqdm

Files:

* data.py
  Loads the data matrix and group labels. Expects a CSV file with shape n×p (rows = observations, columns = variables), skips the first row and first column for headers, and provides:
  X  # numpy array, shape (n, p)
  n  # number of observations
  p  # number of variables
  L  # number of groups
  G  # array of length p, group labels 1..L

* model.py
  Defines:
  • theta: global interaction vector of length L(L-1)/2
  • MLPwithGamma: 3-layer MLP where the first-layer weight is Γ
  • get\_related\_interactions(theta, L, l)

* train.py
  Trains p separate MLPs and learns theta and each model’s gamma via proximal gradient descent with adaptive Lasso and group Lasso penalties.
  Usage:
  python train.py

* result.py
  Builds the p×p adjacency matrix W from the learned theta and gamma weights, applies a threshold of 1e-3, and saves adjacency\_matrix.csv.
  Usage:
  python result.py

Input:

1. data.csv: a CSV file with n rows and p columns of numerical data. The script skips the first row and first column.
2. Group vector G: an array of length p with integer labels 1..L.

Usage:

1. Place your dataset in the project root as data.csv.
2. Edit data.py if your file has a different name or format.
3. Run training:
   python train.py
4. Generate and save the network:
   python result.py
