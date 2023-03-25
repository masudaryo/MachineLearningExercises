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

    alpha = 1e-3
    eta = 0.01

    w = ridge_SGD(*(data_to_X_y(X, Y, 9)), alpha, eta, eps=1e-4, max_epochs=100000)



    x = np.linspace(-0.1, 1.1, 100)
    y = polynomial(w, x)
    fig, ax = plt.subplots()
    ax.scatter(X, Y, label="data")
    ax.plot(x, y, "g", label=fr"$degree = 9, \alpha = {alpha}, \eta = {eta}$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.grid()
    ax.legend()
    plt.show()


