# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


df = pd.read_csv("data.csv", index_col=0)

X = df.to_numpy(dtype=float)

n,p=X.shape

#group assignment vector for 100 variables in 10 groups for synthetic data
G = np.repeat(np.arange(1, 11), 10) 
#group numbers
L = len(np.unique(G))


def get_related_indices(L, l):
    indices = []
    idx = 0
    for i in range(1, L):
        for j in range(i+1, L+1):
            idx += 1
            if i == l or j == l:
                indices.append(idx-1)  
    return indices



