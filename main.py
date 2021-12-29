from math import comb, e
from matplotlib import pyplot as plt
import numpy as np

# CONSTANTS
COOPERATOR = "C"
DEFECTOR = "D"
RICH = "R"
POOR = "P"

# PARAMETERS
# Eperiment of fig. 2.
r = 0.2
Z = 200
Z_R = 40
Z_P = 160
c = 0.1
N = 6
b = 1
M = 3
b_P = 0.625
b_R = 2.5
p_k_ma = [x * 10 ** -3 for x in [2, 40, 75, 3, 2, 20]]
gradient_k_ma = [x * 10 ** -2 for x in [16, 6, 2, 16, 6, 3]]
c_R = c * b_R
c_P = c * b_P
beta = 10
h = 0.0
mu = 1 / Z


def heaviside(k):
    return int(k >= 0)


def payoff(j_R, j_P, X, k):
    if X is DEFECTOR:
        if k is RICH:
            b_strat = b_R
        else:
            b_strat = b_P

        theta = heaviside(c_R * j_R + c_P * j_P - M * c * b)
        return b_strat * (theta + (1 - r) * (1 - theta))
    else:
        if k is RICH:
            c_strat = c_R
        else:
            c_strat = c_P
        return payoff(j_R, j_P, DEFECTOR, k) - c_strat


def fitness(i_R, i_P, X, k):
    if X is COOPERATOR:
        if k is RICH:
            a, b, x, m, n = -1, 0, 0, 1, 0
        else:
            a, b, x, m, n = 0, -1, 0, 0, 1
    else:
        if k is RICH:
            a, b, x, m, n = 0, 0, -1, 0, 0
        else:
            a, b, x, m, n = 0, 0, -1, 0, 0

    sum_ = 0
    for j_R in range(N):
        for j_P in range(N - j_R):
            mul = comb(i_R + a, j_R) * comb(i_P + b, j_P)
            mul *= comb(Z + x - i_R - i_P, N - 1 - j_R - j_P)
            mul *= payoff(j_R + m, j_P + n, X, k)
            sum_ += mul
    return comb(Z - 1, N - 1) ** -1 * sum_


def fermi(i_R, i_P, start_X, end_X, start_k, end_k):
    return 1 / (1 + e ** (beta * (fitness(i_R, i_P, start_X, start_k) - fitness(i_R, i_P, end_X, end_k))))


def T(i_R, i_P, X, Y, k):
    if k is RICH:
        Z_k = Z_R
        Z_l = Z_P
        l = POOR
    else:
        Z_k = Z_P
        Z_l = Z_R
        l = RICH

    if X is COOPERATOR:
        if k is RICH:
            i_X_k = i_R
        else:
            i_X_k = i_P
    else:
        if k is RICH:
            i_X_k = Z_R - i_R
        else:
            i_X_k = Z_P - i_P

    if Y is COOPERATOR:
        if k is RICH:
            i_Y_k = i_R
            i_Y_l = i_P
        else:
            i_Y_k = i_P
            i_Y_l = i_R
    else:
        if k is RICH:
            i_Y_k = Z_R - i_R
            i_Y_l = Z_P - i_P
        else:
            i_Y_k = Z_P - i_P
            i_Y_l = Z_R - i_R

    left_term = i_Y_k / ((Z_k - 1 + (1 - h) * Z_l) * (1 + fermi(i_R, i_P, X, Y, k, k)))
    right_term = (1 - h) * i_Y_l / ((Z_k - 1 + (1 - h) * Z_l) * (1 + fermi(i_R, i_P, X, Y, k, l)))
    return i_X_k / Z * ((1 - mu) * (left_term + right_term) + mu)


def V(i_R, i_P):
    return i_R * Z_P + i_P


def V_minus_1(x):
    return x // Z_P, x % Z_P


def create_transition_matrix(X, Y):
    F = [[0 for _ in range(Z_R)] for _ in range(Z_P)]
    for i_P in range(Z_P):
        for i_R in range(Z_R):
            try:
                F[i_P][i_R] = T(i_R, i_P, X, Y, POOR)
            except ValueError:
                F[i_P][i_R] = 0
    return F


def gradient(i_R, i_P):
    return T(i_R, i_P, DEFECTOR, COOPERATOR, RICH) - T(i_R, i_P, COOPERATOR, DEFECTOR, RICH), \
           T(i_R, i_P, DEFECTOR, COOPERATOR, POOR) - T(i_R, i_P, COOPERATOR, DEFECTOR, POOR)


def distance(x, y):
    return (x ** 2 + y ** 2) ** (1 / 2)


def create_gradient_matrixes():
    X = [[0 for _ in range(Z_R)] for _ in range(Z_P)]
    Y = [[0 for _ in range(Z_R)] for _ in range(Z_P)]
    strength = [[0 for _ in range(Z_R)] for _ in range(Z_P)]
    for i_P in range(Z_P):
        for i_R in range(Z_R):
            try:
                X[i_P][i_R], Y[i_P][i_R] = gradient(i_R, i_P)
                strength[i_P][i_R] = distance(X[i_P][i_R], Y[i_P][i_R])
            except ValueError:
                X[i_P][i_R] = 0
                Y[i_P][i_R] = 0
                strength[i_P][i_R] = 0
    return X, Y, strength


if __name__ == "__main__":
    u, v, strength = create_gradient_matrixes()
    u, v, strength = np.array(u), np.array(v), np.array(strength)
    x, y = np.meshgrid(np.arange(0, Z_R), np.arange(0, Z_P))
    plt.streamplot(x, y, u, v, color=strength, linewidth=1, cmap='jet')
    plt.show()
