import math
import sys

import numpy as np
import pandas as pd
import random
from collections import defaultdict


def run(D, E: np.ndarray, p: int, C: int = 49, shape=(7, 7), n_iter: int = 50, q: int = 5, n: float = 1.1):
    grid = np.random.random((shape[0], shape[1], p))
    x_max = shape[0] - 1
    y_max = shape[1] - 1

    sigma_0 = calculate_initial_radius(x_max, y_max)
    sigma_f = calculate_final_radius()
    delta = calculate_grid_squared_distance_matrix(C, grid, shape)

    # Initialization
    t = 0
    sigma = sigma_0 * (sigma_f / sigma_0) ** (t / (n_iter - 1))

    h = calculate_neighbourhood_function(C, delta, sigma)

    G = init_cluster_representatives(C, E, q)

    V = init_matrix_of_relevance_weigths(C, q)

    P = defaultdict([])
    for k in range(E.shape[0]):
        r = f(E[k])
        P[r].append(E[k])

        def f(elem):
            r = -1
            min_delta_V = sys.float_info.max
            for s in range(C):
                delta_V = calculate_delta_V(elem, s)
                if delta_V < min_delta_V:
                    min_delta_V = delta_V
                    r = s
            return r

        def calculate_delta_V(elem, s):
            delta_V = 0
            for r in range(C):
                delta_V += h[s, r] * D_v_r(elem, G[r])
            return delta_V

        def D_v_r(elem, indices):
            total = 0
            for i in range(len(indices)):
                total += (V[r, i] ** n) * D[k, indices[i]]


def init_matrix_of_relevance_weigths(C, q):
    #: Initial matrix of relevance weights: randomly initialize the matrix V so that for each r sum(v[r,e]) = 1, e in G[r]
    V = []
    for r in range(C):
        values = [random.random() for i in range(q)]
        total = sum(values)
        values = [i / total for i in values]
        V.append(values)
    V = np.array(V)
    return V


def init_cluster_representatives(C, E, q):
    # Initial cluster representatives: randomly select C distinct set-medoids to obtain the initial vector of
    # set-medoids
    G = []
    for r in range(C):
        indices = np.random.choice(E.shape[0], q, replace=False)
        G.append(indices)
    G = np.array(G)
    return G


def calculate_neighbourhood_function(C, delta, sigma):
    # neighbourhood function
    h = np.zeros((C, C))
    for s in range(C):
        for r in range(C):
            h[s, r] = math.exp(-(delta[s][r] / 2 * sigma ** 2))

    return h


def calculate_grid_squared_distance_matrix(C, grid, shape):
    neurons = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            neurons.append({'w': grid[i][j], 'x': i, 'y': j})
    # the squared distance matrix between the nodes of the grid
    delta = np.zeros((C, C))
    for i in range(C):
        for j in range(C):
            if i == j:
                delta[i, j] = 0
            else:
                delta[i, j] = (neurons[i]['x'] - neurons[j]['x']) ** 2 + (neurons[i]['y'] - neurons[j]['y']) ** 2

    return delta


def calculate_final_radius():
    # Also, we compute rf such that two neighboring neurons have a kernel value (hf) equal to 0.01. T
    hf = 0.01
    sigma_f = math.sqrt((-1) / 2 * math.log(hf))
    return sigma_f


def calculate_initial_radius(x_max, y_max):
    # We initialized the radius of the map (r0) that represents the distance of two neurons from a kernel value (h0)
    # equal to 0.1.  The diameter of the map in the topological space is the largest topological distance between two
    # neurons of the map and it is computed from x_max**2 + y_max**2 where xmax and ymax correspond to the size of the
    # grid in the horizontal X-axis and vertical Y-axis, respectively.
    h0 = 0.1
    sigma_0 = math.sqrt((-(x_max ** 2 + y_max ** 2)) / 2 * math.log(h0))
    return sigma_0


if __name__ == '__main__':
    df = pd.read_csv('../../data/HTRU_2.csv', header=None)
    df = df.iloc[:, :-1]
    X = df.to_numpy()
    run(X, 8)
