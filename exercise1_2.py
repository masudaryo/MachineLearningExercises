from mylib import *
import numpy as np
from numpy import newaxis
import matplotlib as mpl
import matplotlib.pyplot as plt


def data_to_X_y(D, degree):
    X = D[:, 0:1] ** np.arange(degree+1)
    y = D[:, 1:2]
    return X, y



if __name__ == '__main__':
    D = np.array([[1, 3], [3, 6], [6, 5], [8, 7]])
    w1 = linear_regression(*(data_to_X_y(D, 1)))[:, 0]
    w2 = linear_regression(*(data_to_X_y(D, 2)))[:, 0]
    w3 = linear_regression(*(data_to_X_y(D, 3)))[:, 0]

    eps = D[:, 1] - polynomial(w1, D[:, 0])
    print(f"単回帰の係数: a = {w1[1]}, b = {w1[0]}")
    print("各事例の残差:", eps)
    print("説明変数と残差の共分散:", cov(D[:, 0], eps))
    print("目的変数の推定値と残差の共分散:", cov(polynomial(w1, D[:, 0]), eps))


    print(f"1次の回帰の決定係数: {coef_of_determination(D[:, 1], polynomial(w1, D[:, 0]))}")
    print(f"2次の回帰の決定係数: {coef_of_determination(D[:, 1], polynomial(w2, D[:, 0]))}")
    print(f"3次の回帰の決定係数: {coef_of_determination(D[:, 1], polynomial(w3, D[:, 0]))}")


    x = np.linspace(0, 10, 100)
    y1 = polynomial(w1, x)
    y2 = polynomial(w2, x)
    y3 = polynomial(w3, x)
    fig, ax = plt.subplots()
    ax.scatter(D[:, 0], D[:, 1], label="data")
    ax.plot(x, y1, "g", label=r"$y=w_0 + w_1 x$")
    ax.plot(x, y2, "r", label=r"$y=w_0 + w_1 x + w_2 x ^ 2$")
    ax.plot(x, y3, "c", label=r"$y=w_0 + w_1 x + w_2 x ^ 2 + w_3 x ^ 3$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.grid()
    ax.legend()
    plt.show()





