from math import comb, e
from matplotlib import pyplot as plt
import numpy as np

# CONSTANTS
COOPERATOR = "C"
DEFECTOR = "D"
RICH = "R"
POOR = "P"


class Project:

    def __init__(self):

        # PARAMETERS
        self.r = 0.3  # risk perception (in [0, 1])
        self.Z_R = 10
        self.Z_P = 10
        self.Z = self.Z_R + self.Z_P
        self.c = 0.1  # fraction of endowment used to solve the group task
        self.N = 6  # groups size

        self.b_R = 1.35
        self.b_P = 1 / self.Z_P * (self.Z - self.b_R * self.Z_R)
        self.b = (self.b_R * self.Z_R + self.b_P * self.Z_P) / (self.Z_R + self.Z_P)  # average of endowments

        self.M = 3  # * c * b  # parameter in ]0, N] but not sure what it exactly represents

        self.c_R = self.c * self.b_R
        self.c_P = self.c * self.b_P
        self.beta = 5  # controls the intensity of selection
        self.h = 1  # homophily
        self.mu = 1 / self.Z  # mutation probability

    @staticmethod
    def heaviside(k):
        return int(k >= 0)

    def payoff(self, j_R, j_P, X, k):
        """
        :param j_R: number of rich cooperators
        :param j_P: number of poor cooperators
        :param X: cooperator/defector
        :param k: rich/poor
        :return: payoff for strategy (X, k)
        """
        if X == DEFECTOR:
            if k == RICH:
                b_strat = self.b_R
            else:
                b_strat = self.b_P
            theta = Project.heaviside(self.c_R * j_R + self.c_P * j_P - self.M * self.c * self.b)
            return b_strat * (theta + (1 - self.r) * (1 - theta))

        else:
            if k == RICH:
                c_strat = self.c_R
            else:
                c_strat = self.c_P
            return self.payoff(j_R, j_P, DEFECTOR, k) - c_strat

    def fitness(self, i_R, i_P, X, k):
        if X == COOPERATOR:
            if k == RICH:
                a, b, x, m, n = -1, 0, 0, 1, 0
            else:
                a, b, x, m, n = 0, -1, 0, 0, 1
        else:
            if k == RICH:
                a, b, x, m, n = 0, 0, -1, 0, 0
            else:
                a, b, x, m, n = 0, 0, -1, 0, 0

        sum_ = 0
        for j_R in range(self.N):
            for j_P in range(self.N - j_R):
                if i_R + a >= 0:
                    mul = comb(i_R + a, j_R)
                else:
                    mul = comb(0, j_R)
                if i_P + b >= 0:
                    mul *= comb(i_P + b, j_P)
                else:
                    mul *= comb(0, j_P)
                mul *= comb(self.Z + x - i_R - i_P, self.N - 1 - j_R - j_P)
                mul *= self.payoff(j_R + m, j_P + n, X, k)
                sum_ += mul
        return comb(self.Z - 1, self.N - 1) ** -1 * sum_

    def fermi(self, i_R, i_P, start_X, end_X, start_k, end_k):
        return (1 + e ** (self.beta *
                          (self.fitness(i_R, i_P, start_X, start_k) - self.fitness(i_R, i_P, end_X, end_k)))) ** -1

    def T(self, i_R, i_P, X, Y, k):
        if k == RICH:
            Z_k = self.Z_R
            Z_l = self.Z_P
            l = POOR
        else:
            Z_k = self.Z_P
            Z_l = self.Z_R
            l = RICH

        if X == COOPERATOR:
            if k == RICH:
                i_X_k = i_R
            else:
                i_X_k = i_P
        else:
            if k == RICH:
                i_X_k = self.Z_R - i_R
            else:
                i_X_k = self.Z_P - i_P

        if Y == COOPERATOR:
            if k == RICH:
                i_Y_k = i_R
                i_Y_l = i_P
            else:
                i_Y_k = i_P
                i_Y_l = i_R
        else:
            if k == RICH:
                i_Y_k = self.Z_R - i_R
                i_Y_l = self.Z_P - i_P
            else:
                i_Y_k = self.Z_P - i_P
                i_Y_l = self.Z_R - i_R

        left_term = i_Y_k / (Z_k - 1 + (1 - self.h) * Z_l) * self.fermi(i_R, i_P, X, Y, k, k)
        right_term = (1 - self.h) * i_Y_l / (Z_k - 1 + (1 - self.h) * Z_l) * self.fermi(i_R, i_P, X, Y, k, l)
        return (i_X_k / self.Z) * ((1 - self.mu) * (left_term + right_term) + self.mu)

    # def T_transition(self, i_R, i_P, i_R_prime, i_P_prime):
    #
    #     if i_R == i_R_prime and i_P == i_P_prime:
    #         return self.T(i_R, i_P, COOPERATOR, COOPERATOR, RICH) * self.T(i_R, i_P, COOPERATOR, COOPERATOR, POOR)
    #     elif i_R == i_R_prime and i_P < i_P_prime:
    #         return self.T(i_R, i_P, COOPERATOR, COOPERATOR, RICH) * self.T(i_R, i_P, DEFECTOR, COOPERATOR, POOR)
    #     elif i_R == i_R_prime and i_P > i_P_prime:
    #         return self.T(i_R, i_P, COOPERATOR, COOPERATOR, RICH) * self.T(i_R, i_P, COOPERATOR, DEFECTOR, POOR)
    #
    #     elif i_R < i_R_prime and i_P == i_P_prime:
    #         return self.T(i_R, i_P, DEFECTOR, COOPERATOR, RICH) * self.T(i_R, i_P, COOPERATOR, COOPERATOR, POOR)
    #     elif i_R < i_R_prime and i_P < i_P_prime:
    #         return self.T(i_R, i_P, DEFECTOR, COOPERATOR, RICH) * self.T(i_R, i_P, DEFECTOR, COOPERATOR, POOR)
    #     elif i_R < i_R_prime and i_P > i_P_prime:
    #         return self.T(i_R, i_P, DEFECTOR, COOPERATOR, RICH) * self.T(i_R, i_P, COOPERATOR, DEFECTOR, POOR)
    #
    #     elif i_R > i_R_prime and i_P == i_P_prime:
    #         return self.T(i_R, i_P, COOPERATOR, DEFECTOR, RICH) * self.T(i_R, i_P, COOPERATOR, COOPERATOR, POOR)
    #     elif i_R > i_R_prime and i_P < i_P_prime:
    #         return self.T(i_R, i_P, COOPERATOR, DEFECTOR, RICH) * self.T(i_R, i_P, DEFECTOR, COOPERATOR, POOR)
    #     elif i_R > i_R_prime and i_P > i_P_prime:
    #         return self.T(i_R, i_P, COOPERATOR, DEFECTOR, RICH) * self.T(i_R, i_P, COOPERATOR, DEFECTOR, POOR)
    #
    #     else:
    #         print("ERROR")
    #
    # def create_matrix_W(self):
    #     size = self.Z_R * self.Z_P
    #     W = [[0 for _ in range(size)] for _ in range(size)]
    #     for i_R in range(self.Z_R):
    #         for i_P in range(self.Z_P):
    #             print("i_R: {0}, i_P: {1}".format(i_R, i_P))
    #             p = self.V(i_R, i_P)
    #             for i_R_prime in range(self.Z_R):
    #                 for i_P_prime in range(self.Z_P):
    #                     q = self.V(i_R_prime, i_P_prime)
    #                     W[p][q] = self.T_transition(i_R, i_P, i_R_prime, i_P_prime)
    #                     # print("i_R: {0}, i_P: {1}, i_R_prime: {2}, i_P_prime: {3}, W[p][q]: {4}".format(i_R, i_P, i_R_prime, i_P_prime, W[p][q]))
    #     plt.matshow(np.array(W))
    #     plt.show()
    #     return W
    #
    # def get_probabilities(self, W):
    #     eigenvalues, eigenvectors = np.linalg.eig(W)
    #     print(np.linalg.eig(W))
    #     for i in range(len(eigenvalues)):
    #         if eigenvalues[i] == 1.0:
    #             proba = [[0 for _ in range(self.Z_R)] for _ in range(self.Z_P)]
    #             for j in range(len(eigenvectors[i])):
    #                 i_R, i_P = self.V_minus_1(j)
    #                 proba[i_R][i_P] = float(eigenvectors[i][j])
    #             plt.matshow(np.array(proba))
    #             plt.show()
    #             return proba
    #     print("Eigenvalue 1 not found")
    #
    # def V(self, i_R, i_P):
    #     return i_R * self.Z_P + i_P
    #
    # def V_minus_1(self, x):
    #     return x // self.Z_P, x % self.Z_P

    def gradient(self, i_R, i_P):
        return self.T(i_R, i_P, DEFECTOR, COOPERATOR, RICH) - self.T(i_R, i_P, COOPERATOR, DEFECTOR, RICH), \
               self.T(i_R, i_P, DEFECTOR, COOPERATOR, POOR) - self.T(i_R, i_P, COOPERATOR, DEFECTOR, POOR)

    @staticmethod
    def distance(x, y):
        return (x ** 2 + y ** 2) ** (1 / 2)

    def create_gradient_matrices(self):
        X = [[0 for _ in range(self.Z_R)] for _ in range(self.Z_P)]
        Y = [[0 for _ in range(self.Z_R)] for _ in range(self.Z_P)]
        strength = [[0 for _ in range(self.Z_R)] for _ in range(self.Z_P)]
        for i_P in range(self.Z_P):
            for i_R in range(self.Z_R):
                X[i_P][i_R], Y[i_P][i_R] = self.gradient(i_R, i_P)
                strength[i_P][i_R] = Project.distance(X[i_P][i_R], Y[i_P][i_R])
        return X, Y, strength

    def plotGradients(self):
        # set parameters of Figure 1 (SI) :
        self.N = 10
        self.M = 3
        self.beta = 10
        self.h = 0
        self.r = 0.3
        self.c = 0.1
        self.c_R = self.c * self.b_R
        self.c_P = self.c * self.b_P

        fig, axs = plt.subplots(2, 2, figsize=(10, 5))
        ZR = self.Z_R
        ZP = self.Z_P

        self.plotGrad(axs[0, 0], RICH, 100, 100)
        self.plotGrad(axs[1, 0], POOR, 100, 100)
        self.plotGrad(axs[0, 1], RICH, 40, 160)
        self.plotGrad(axs[1, 1], POOR, 40, 160)

        axs[0, 0].set_title(r"$Z_P = Z_R$")
        axs[0, 1].set_title(r"$Z_P = 4Z_R$")
        axs[0, 0].set(ylabel=r"$∇_i$ (Rich)")
        axs[1, 0].set(ylabel=r"$∇_i$ (Poor)")
        axs[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        axs[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        # for ax in axs.flat:
        #    ax.label_outer()

        fig.tight_layout()  # avoid the superpose the different plots
        plt.show()
        self.Z_R = ZR
        self.Z_P = ZP

    def plotGrad(self, ax, X, ZR, ZP):
        self.Z_R = ZR
        self.Z_P = ZP
        bs = [1.35, 1.75]
        line = ["--", ""]
        fraction = [0.9, 0.5, 0.1]
        colors = ["green", "orange", "blue"]
        for b in range(len(bs)):
            self.b_R = bs[b]
            self.b_P = self.b_P = 1 / self.Z_P * (self.Z - self.b_R * self.Z_R)
            self.b = (self.b_R * self.Z_R + self.b_P * self.Z_P) / (self.Z_R + self.Z_P)
            if X == RICH:

                grad = [[0 for _ in range(self.Z_R)] for _ in range(len(fraction))]
                for frac in range(len(fraction)):
                    i_P = int(round(self.Z_P * fraction[frac], 0))
                    for i_R in range(1, self.Z_R):
                        grad[frac][i_R] = self.gradient(i_R, i_P)
                    x = np.arange(0 + 1 / self.Z_R, 1, 1 / self.Z_R)
                    y = [i[0] for i in grad[frac][1:self.Z_R]]

                    ax.plot(x, y, line[b], color=colors[frac],
                            label=str(round(fraction[frac] * 100, 0)) + " % * " + r"$Z_R$" + " ; " + r"$b_R$" + " = " + str(bs[b]))
                    ax.set(xlabel=r"$i_R/Z_R$")

            if X == POOR:
                x = np.arange(0 + 1 / self.Z_P, 1, 1 / self.Z_P)
                grad = [[0 for _ in range(self.Z_P)] for _ in range(len(fraction))]
                for frac in range(len(fraction)):
                    i_R = int(round(self.Z_R * fraction[frac], 0))
                    for i_P in range(1, self.Z_P):
                        grad[frac][i_P] = self.gradient(i_R, i_P)
                    y = [i[1] for i in grad[frac][1:self.Z_P]]
                    ax.plot(x, y, line[b], color=colors[frac],
                            label=str(round(fraction[frac] * 100, 0)) + " % * " + r"$Z_R$" + " ; " + r"$b_R$" + " = " + str(bs[b]))
                    ax.set(xlabel=r"$i_P/Z_P$")

    def plotOneGraphFig2(self, ax):
        u, v, strength = self.create_gradient_matrices()
        u, v, strength = np.array(u), np.array(v), np.array(strength)
        x, y = np.meshgrid(np.arange(0, self.Z_R), np.arange(0, self.Z_P))

        strm = ax.streamplot(x, y, u, v, color=strength, linewidth=0.5, arrowsize=0.5, density=1, cmap='jet')
        ax.set(xlabel=r"$i_R$", ylabel=r"$i_P$")
        ax.set_title("h=" + str(self.h))
        return strm

    def plotMultiFig(self):
        fig, axs = plt.subplots(2, 3, figsize=(10, 5))
        pad = 5  # used for plot label
        homophilies = [0, 0.7, 1]
        risks = [0.2, 0.3]
        for r in range(len(risks)):
            self.r = risks[r]
            for h in range(len(homophilies)):
                self.h = homophilies[h]
                strm = self.plotOneGraphFig2(axs[r, h])
        # fig.colorbar(strm.lines, shrink=0.5, label="Gradient of selection ∇") #need to find how to fix the position
        axs[0, 0].annotate("r = 0.2", xy=(0, 0.5), xytext=(-axs[0, 0].yaxis.labelpad - pad, 0),
                           xycoords=axs[0, 0].yaxis.label, textcoords='offset points',
                           size='large', ha='right', va='center')
        axs[1, 0].annotate("r = 0.3", xy=(0, 0.5), xytext=(-axs[1, 0].yaxis.labelpad - pad, 0),
                           xycoords=axs[1, 0].yaxis.label, textcoords='offset points',
                           size='large', ha='right', va='center')
        fig.tight_layout()  # avoid the superpose the different plots
        plt.show()

    def set_parameters(self, Z_R, Z_P, b_R, c=0.1, N=6, M=3, beta=5):
        self.Z_R = Z_R
        self.Z_P = Z_P
        self.Z = self.Z_R + self.Z_P
        self.c = c
        self.N = N
        self.b_R = b_R
        self.b_P = 1 / self.Z_P * (self.Z - self.b_R * self.Z_R)
        self.b = (self.b_R * self.Z_R + self.b_P * self.Z_P) / (self.Z_R + self.Z_P)
        self.M = M
        self.c_R = self.c * self.b_R
        self.c_P = self.c * self.b_P
        self.beta = beta
        self.mu = 1 / self.Z

    def plotFig2SI(self):
        # set parameters of figure 2 (SI) :
        self.set_parameters(Z_R=100, Z_P=100, b_R=1.7)
        self.plotMultiFig()

    def plotFig2Paper(self):
        # set parameters of figure 2 (Paper) :
        self.set_parameters(Z_R=40, Z_P=160, b_R=2.5)
        self.plotMultiFig()


if __name__ == "__main__":
    project = Project()
    project.plotGradients()
    project.plotFig2SI()
    project.plotFig2Paper()
