from mylib import *
import numpy as np
from numpy import newaxis
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot(ax, f):
    X = np.linspace(-4.5, 3.5, 1000)
    Y = f(X)

    ax.plot(X, Y, '-')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal')

def F(v, w, b, x):
    return sigmoid(x[:, newaxis] @ w[newaxis, :] + b) @ v





if __name__ == '__main__':
    w = np.array([-1000] + [1000]*7)
    x = np.arange(-4, 4)
    b = -w * x
    v = np.array([-1, 1, -1, -1, 2, -1, 1, -2])


    fig, ax = plt.subplots()
    X = np.linspace(-4.5, 3.5, 1000)
    Y = F(v, w, b, X)

    ax.plot(X, Y, '-')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal')
    plt.show()
