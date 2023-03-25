from mylib import *
import numpy as np
from numpy import newaxis

if __name__ == '__main__':
    X = np.array([[-7, -2], [-3, -3], [4, 1], [6, 4]])

    S = X.T @ X
    print(f"{S = }")
    w, v = np.linalg.eig(S)
    print(f"固有値: {w}, 固有ベクトル: {v[:, 0]}, {v[:, 1]}")
    temp = np.flip(np.argsort(w))
    for i in range(len(temp)):
        print(f"第{i+1}主成分: {v[:, temp[i]]}")
        print(f"分散: {var(X @ v[:, temp[i]])}")
        print(f"第{i+1}主成分得点: {X @ v[:, temp[i]]}")
        print()


