import math
import multiprocessing
import pickle
import random
import sys
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from scipy.special import binom

from dissimilarity import calculate_dissimilarity_matrix

# Neighbourhood matrix
H: np.ndarray

# Cluster representatives matrix
G: np.array

# Matrix of relevance weights
V: np.ndarray

# Partition
P: defaultdict
F: np.array


def run(D: np.ndarray, E: np.ndarray, Y: np.ndarray, C: int = 36, shape=(6, 6), n_iter: int = 50, q: int = 5,
        n: float = 1.1):
    """

    :param D: Dissimilarity matrix
    :param E: Set of elements
    :param C: Number of neurons in the SOM
    :param shape: SOM (map) shape
    :param n_iter: number of iterations
    :param q: cardinatilty for each set of medoids
    :param n: smoothing parameter
    :return:
    """
    global F

    before = time.time()
    x_max = shape[0]
    y_max = shape[1]

    sigma_0 = calculate_initial_radius(x_max, y_max)
    sigma_f = calculate_final_radius()
    delta, neurons = calculate_grid_squared_distance_matrix(C, shape)

    # Initialization
    t = 0
    sigma = sigma_0

    # init H
    calculate_neighbourhood_function(C, delta, sigma)

    # init G
    init_cluster_representatives(C, E, q)

    # init V
    init_matrix_of_relevance_weigths(C, q)

    # Initial assignment: obtain the initial partition
    F = np.zeros(E.shape[0], dtype=object)
    update_assignment(E, n, D, C)

    N = E.shape[0]

    while t < n_iter:
        if t < 10:
            print('V Matrix at iter ' + str(t) + ':')
            print(V)
        t += 1
        sigma = sigma_0 * ((sigma_f / sigma_0) ** (t / (n_iter - 1)))
        calculate_neighbourhood_function(C, delta, sigma)

        # Step 1: representation: compute the elements of the vector of set-medoids.
        # During the representation step, the matrix V and the partition P are kept fixed.  The cost function is
        # minimized with respect to the vector of prototypes G.
        update_set_medoids(C, N, D, q)

        # Step 2: weighting: compute the elements v[r,e] of the matrix of weights V
        update_matrix_of_relevance_weights(N, n, D)

        # Step 3: assignment: obtain the partition
        update_assignment(E, n, D, C)

        # Calculates quantization error
        QE = calculate_quantization_error(N, n, D)
        print('QE = ' + str(QE))

        TE = calculate_topological_error(N, neurons)
        print('TE = ' + str(TE))

        contingency_matrix, n_i, n_j, num_classes = calculate_contingency_matrix(C, Y)
        ARI = calculate_adjusted_rand_index(C, N, contingency_matrix, n_i, n_j, num_classes)
        print('ARI = ' + str(ARI))

        F_measure = calculate_F_measure(C, N, contingency_matrix, n_i, n_j, num_classes)
        print('F-measure: ' + str(F_measure))

        if t >= n_iter - 10:
            print('V Matrix at iter ' + str(t) + ':')
            print(V)

        print('Objective function value:')
        J = obj_function(N, C, n, D)
        print(J)

    print('G Matrix:')
    print(G)
    print('Confusion matrix:')
    print(contingency_matrix)
    sn.heatmap(contingency_matrix, annot=True)
    plt.show()

    print('Final objective function value:')
    J = obj_function(N, C, n, D)
    print(J)

    after = time.time()
    print(str(after - before) + ' seconds.')


def obj_function(N, C, n, D):
    def delta_v(k):
        total = 0
        for r in range(C):
            total += H[F[k][0], r] * D_v_r(k, G[r], n, r, D, V)
        return total

    J = 0
    for k in range(N):
        J += delta_v(k)
    return J


def calculate_F_measure(C, N, contingency_matrix, n_i, n_j, num_classes):
    F_measure = 0
    for j in range(num_classes):
        max_i = sys.float_info.min
        for i in range(C):
            if n_i[i] != 0 and n_j[j] != 0 and contingency_matrix[i, j] != 0:
                max_calc = ((contingency_matrix[i, j] / n_i[i]) * contingency_matrix[i, j] / n_j[j]) / (
                        (contingency_matrix[i, j] / n_i[i]) + (contingency_matrix[i, j] / n_j[j]))
                if max_calc > max_i:
                    max_i = max_calc
        F_measure += (n_j[j] / N) * max_i

    return F_measure


def calculate_adjusted_rand_index(C, N, contingency_matrix, n_i, n_j, num_classes):
    # Calculates Adjusted Rand Index
    m = 0
    for i in range(C):
        for j in range(num_classes):
            m += binom(contingency_matrix[i, j], 2)
    m1 = 0
    for i in range(C):
        m1 += binom(n_i[i], 2)
    m2 = 0
    for j in range(num_classes):
        m2 += binom(n_j[j], 2)
    M = binom(N, 2)
    ARI = (m - (m1 * m2) / M) / (m1 / 2 + m2 / 2 - (m1 * m2) / M)
    return ARI


def calculate_contingency_matrix(C, Y):
    num_classes = len(np.unique(Y))
    contingency_matrix = np.zeros((C, num_classes))
    for i in range(C):
        for j in range(num_classes):
            P_i = set(P[i])
            C_j = set((np.where(Y == j)[0].flatten()))
            contingency_matrix[i, j] = len(P_i.intersection(C_j))
    n_i = np.sum(contingency_matrix, axis=1)
    n_j = np.sum(contingency_matrix, axis=0)
    return contingency_matrix, n_i, n_j, num_classes


def calculate_topological_error(N, neurons):
    def u(k):
        first_r, second_r = F[k]
        first_x = neurons[first_r]['x']
        first_y = neurons[first_r]['y']
        second_x = neurons[second_r]['x']
        second_y = neurons[second_r]['y']
        if abs(first_x - second_x) <= 1 and abs(first_y - second_y) <= 1:
            return 0
        return 1

    TE = 0
    for k in range(N):
        TE += u(k)
    TE = TE / N
    return TE


def calculate_quantization_error(N, n, D):
    QE = 0
    for k in range(N):
        # bmu = BMU(E, F[k][0])
        # QE += LA.norm(E[k] - bmu) ** 2
        QE += (D_v_r(k, G[F[k][0]], n, F[k][0], D, V)) ** 2
    QE = QE / N

    return QE


def update_set_medoids(C, N, D, q):
    global G
    global H
    global F

    before = time.time()
    procs = C
    jobs = []
    for i in range(procs):
        process = multiprocessing.Process(target=update_set_medoids_for_cluster, args=(D, G, N, q, i, H, F))
        jobs.append(process)

    # Start the threads
    for j in jobs:
        j.start()

    # Ensure all of the threads have finished
    for j in jobs:
        j.join()

    after = time.time()
    print('Set-medoids updating complete.  Time elapsed = ' + str(after - before))


def update_set_medoids_for_cluster(D, G, N, q, r, H, F):
    g = np.zeros(N)
    # print('Updating medoids for cluster ' + str(r) + '...')
    before = time.time()
    for h in range(N):
        for k in range(N):
            g[h] += H[F[k][0], r] * D[k, h]
            # g[h] += H[f(k, n, D, C, H, G, V)[0], r] * D[k, h]
    indices = np.argsort(g)[:q]
    G[r] = indices
    after = time.time()
    # print('Time spent: ' + str(after - before) + ' seconds.')


def update_matrix_of_relevance_weights(N, n, D):
    for r in range(V.shape[0]):
        for e in range(V.shape[1]):
            # numerator
            num_total = 0
            for k in range(N):
                num_total += H[F[k][0], r] * D[k, G[r, e]]
                # num_total += H[f(k, n, D, C, H, G, V)[0], r] * D[k, G[r, e]]

            total_l = 0

            # denominator
            for l in G[r]:
                den_total = 0
                for k in range(N):
                    den_total += H[F[k][0], r] * D[k, l]

                total_l += (num_total / den_total) ** (1 / (n - 1))

            V[r, e] = 1 / total_l


def f(k, n, D, C, H, G, V):
    deltas = np.zeros(C)
    for s in range(C):
        deltas[s] = calculate_delta_V(k, s, n, D, C, H, G, V)

    sorted_deltas = deltas.argsort()
    r = sorted_deltas[0]
    second_r = sorted_deltas[1]
    return r, second_r


def calculate_delta_V(k, s, n, D, C, H, G, V):
    delta_V = 0
    for r in range(C):
        delta_V += H[s, r] * D_v_r(k, G[r], n, r, D, V)
    return delta_V


def D_v_r(k, Gr, n, r, D, V):
    total = 0
    for i in range(len(Gr)):
        total += (V[r, i] ** n) * D[k, Gr[i]]
    return total


def update_assignment(E, n, D, C):
    # During the assignment step, the vector of prototypes G and the matrix of relevance weights V are kept fixed.  The
    # aim is to minimize the error function with respect to the partition P. The error function is minimized if for each
    # e(k) in E, delta_V(e(k), G(f(e(k))) is minimized.  For a fixed vector or prototypes P and a fixed matrix of
    # relevance weights V, delta_V(e(k), G(f(e(k))) is minimized if f(e(k)) = argmin(1<=s<=C)(delta_V(e(k), G(s)).
    global P
    global F
    P = defaultdict(list)
    for k in range(E.shape[0]):
        r, second_r = f(k, n, D, C, H, G, V)
        # P[r].append(E[k])
        P[r].append(k)
        F[k] = (r, second_r)


def init_matrix_of_relevance_weigths(C, q):
    #: Initial matrix of relevance weights: randomly initialize the matrix V so that for each r sum(v[r,e]) = 1, e in G[r]
    global V
    V = []
    for r in range(C):
        values = [random.random() for _ in range(q)]
        total = sum(values)
        values = [i / total for i in values]
        V.append(values)
    V = np.array(V)


def init_cluster_representatives(C, E, q):
    # Initial cluster representatives: randomly select C distinct set-medoids to obtain the initial vector of
    # set-medoids
    global G
    G = []
    indices = np.random.choice(E.shape[0], q * C, replace=False)
    i = 0
    for r in range(C):
        G.append(indices[i:i + q])
        i += q
    G = np.array(G)


def calculate_neighbourhood_function(C, delta, sigma):
    # neighbourhood function
    global H
    H = np.zeros((C, C))
    for s in range(C):
        for r in range(C):
            H[s, r] = math.exp(-(delta[s, r] / (2 * (sigma ** 2))))


def calculate_grid_squared_distance_matrix(C, shape):
    neurons = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            neurons.append({'x': i, 'y': j})
    # the squared distance matrix between the nodes of the grid
    delta = np.zeros((C, C))
    for i in range(C):
        for j in range(C):
            if i == j:
                delta[i, j] = 0
            else:
                delta[i, j] = (neurons[i]['x'] - neurons[j]['x']) ** 2 + (neurons[i]['y'] - neurons[j]['y']) ** 2

    return delta, neurons


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


def normalize(dist_mat):
    # Each dissimilarity d(e(k), e(l)) (1 <= k,l <= N) in a given dissimilarity matrix D is nomralized as
    # d(e(k), e(l))/T, where T = sum(d(e(k), g)) is the overall dispersion and g = e(l) in E = {e(1),..., e(N)} is the
    # overall representative, which is computed according to l = argmin(1 <= h <= N)(sum(d(e(k), e(h))).
    l = np.sum(dist_mat, axis=0).argmin()
    T = np.sum(dist_mat[:, l])
    dist_mat = dist_mat / T

    # Observe that after normalizing D, we have T = 1.
    l = np.sum(dist_mat, axis=0).argmin()
    T = np.sum(dist_mat[:, l])
    assert T == 1

    return dist_mat


if __name__ == '__main__':
    df = pd.read_csv('../../data/HTRU_2.csv', header=None)

    # Gets aprox. 10% from the positive/negative examples, respectively
    y_column = len(df.columns) - 1
    df_pos = df[df[y_column] == 1]
    df_neg = df[df[y_column] == 0]
    df_pos = df_pos.sample(round(df_pos.shape[0] / 10), random_state=42)
    df_neg = df_neg.sample(round(df_neg.shape[0] / 10), random_state=42)
    df = pd.concat([df_pos, df_neg])
    df = df.sample(frac=1).reset_index(drop=True)

    Y = df.iloc[:, -1]
    df = df.iloc[:, :-1]
    X = df.to_numpy()
    dist_mat = None

    try:
        with open('D.pickle', 'rb') as infile:
            dist_mat = pickle.load(infile)
    except FileNotFoundError:
        dist_mat = calculate_dissimilarity_matrix(X)
        dist_mat = normalize(dist_mat)

        # Saves dissimilarity matrix
        with open('D.pickle', 'wb') as outfile:
            pickle.dump(dist_mat, outfile)

    run(dist_mat, X, Y)
