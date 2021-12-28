from math import comb

# CONSTANTS
COOPERATOR = "C"
DEFECTOR = "D"
RICH = "R"
POOR = "P"

# PARAMETERS
# Experiment of fig. 2.
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
p_k_max = [x * 10 ** -3 for x in [2, 40, 75, 3, 2, 20]]
gradient_k_max = [x * 10 ** -2 for x in [16, 6, 2, 16, 6, 3]]
c_R = c * b_R
c_P = c * b_P


def heaviside(k):
    return int(k >= 0)


def payoff(j_R, j_P, strategy, wealth):
    if strategy is DEFECTOR:
        if wealth is RICH:
            b_strat = b_R
        else:
            b_strat = b_P

        theta = heaviside(c_R*j_R + c_P*j_P - M*c*b)
        return b_strat * (theta + (1 - r) * (1 - theta))
    else:
        if wealth is RICH:
            c_strat = c_R
        else:
            c_strat = c_P
        return payoff(j_R, j_P, DEFECTOR, wealth) - c_strat


def fitness(i_R, i_P, strategy, wealth):
    if strategy is COOPERATOR:
        if wealth is RICH:
            a, b, x, m, n = -1, 0, 0, 1, 0
        else:
            a, b, x, m, n = 0, 0, -1, 0, 0
    else:
        if wealth is RICH:
            a, b, x, m, n = 0, -1, 0, 0, 1
        else:
            a, b, x, m, n = 0, 0, -1, 0, 0

    sum_ = 0
    for j_R in range(N):
        for j_P in range(N - j_R):
            mul = comb(i_R + a, j_R) * comb(i_P + b, j_P)
            mul *= comb(Z + x - i_R - i_P, N - 1 - j_R - j_P)
            mul *= payoff(j_R + m, j_P + n, strategy, wealth)
            sum_ += mul
    return comb(Z - 1, N - 1)**-1 * sum_


if __name__ == "__main__":
    pass
