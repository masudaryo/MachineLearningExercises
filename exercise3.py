from mylib import *
import numpy as np
from numpy import newaxis
import matplotlib as mpl
import matplotlib.pyplot as plt



def data_to_X_y(X, Y, degree):
    return X[:, newaxis] ** np.arange(degree+1), Y



if __name__ == '__main__':
    X = np.array([ 0.  ,  0.16,  0.22,  0.34,  0.44,  0.5 ,  0.67,  0.73,  0.9 ,  1.  ])
    Y = np.array([-0.06,  0.94,  0.97,  0.85,  0.25,  0.09, -0.9 , -0.93, -0.53,  0.08])
    alphas = np.array([10**(-9), 10**(-6), 10**(-3), 1])
    w = ridge_regression(*(data_to_X_y(X, Y, 9)), alphas)

    l2_norm = np.linalg.norm(w, axis=1)
    print("L_2ノルム")
    print(f"alpha = 10^-9: {l2_norm[0]}")
    print(f"alpha = 10^-6: {l2_norm[1]}")
    print(f"alpha = 10^-3: {l2_norm[2]}")
    print(f"alpha = 1: {l2_norm[3]}")

    X_valid = np.array([ 0.05,  0.08,  0.12,  0.16,  0.28,  0.44,  0.47,  0.55,  0.63,  0.99])
    Y_valid = np.array([ 0.35,  0.58,  0.68,  0.87,  0.83,  0.45,  0.01, -0.36, -0.83, -0.06])


    temp = alphas[np.argmin(np.sum((polynomial(w.T, X_valid).T - Y_valid) ** 2, axis=1) / X_valid.size)]
    print("最も汎化性能が高い正則化パラメータ:", temp)


    x = np.linspace(-0.1, 1.1, 100)
    y = polynomial(w.T, x).T
    fig, ax = plt.subplots()
    ax.scatter(X, Y, label="data")
    ax.scatter(X_valid, Y_valid, label="validation data")

    ax.plot(x, y[0], "g", label=r"$\alpha = 10^{-9}$")
    ax.plot(x, y[1], "r", label=r"$\alpha = 10^{-6}$")
    ax.plot(x, y[2], "c", label=r"$\alpha = 10^{-3}$")
    ax.plot(x, y[3], "m", label=r"$\alpha = 1$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.grid()
    ax.legend()
    plt.show()










