import numpy as np
from scipy.spatial import distance_matrix
import pandas as pd
import time

def calculate_dissimilarity_matrix(X):
    dist_mat = distance_matrix(X, X, p=2)
    return dist_mat


if __name__ == '__main__':
    df = pd.read_csv('../../data/HTRU_2.csv', header=None)
    df = df.iloc[: , :-1]
    X = df.to_numpy()
    print(X.shape)

    before = time.time()
    dist_mat = calculate_dissimilarity_matrix(X)
    after = time.time()
    print(dist_mat)
    print(after - before)
