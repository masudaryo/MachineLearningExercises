import numpy as np
from numpy import newaxis

# coef: 1-D array, 多項式の係数. coef.size-1が次数となる。
# x: 1-D array, xの全ての要素に対する多項式の値のndarrayを返す。
# return size が x.size の 1-D array
# coefが 2-D array のとき、coefの各列を多項式の係数とする。
# return 2-D array, shape は (x.size, coef.shape[1])
def polynomial(coef, x):
    return (x[:, newaxis] ** np.arange(len(coef))) @ coef

# 式(2.25)
def linear_regression(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y


# X: 1-D array
def var(X):
    return np.sum((X - np.sum(X)/X.size)**2)/X.size

# Y_d: 1-D array, データの分布
# Y_r: 1-D array, 回帰で得た値
def coef_of_determination(Y_d, Y_r):
    return var(Y_r)/var(Y_d)

# X, Y: 1-D array
def cov(X, Y):
    return np.sum((X - np.sum(X)/X.size) * (Y - np.sum(X)/X.size))/X.size


# 式(3.5)
# X: 2-D array. y: 1-D array. alphas: 1-D array
def ridge_regression(X, y, alphas):
    return np.linalg.inv(X.T @ X + alphas[:, newaxis, newaxis] * np.identity(y.size)) @ X.T @ y


def ridge_SGD(X, y, alpha, eta, eps=10**(-4), max_epochs=10000, init_w=None):
    if init_w is None:
        w = np.zeros((X.shape[1]))
    else:
        w = init_w

    rng = np.random.default_rng()
    for _ in range(max_epochs):
        i = rng.integers(0, X.shape[1])
        grad = -2 * X[i] * (y[i] - X[i] @ w) + 2 * alpha / X.shape[0] * w
        if np.linalg.norm(grad, ord=1) < eps:
            print(f"The grad became less than eps at the epoch {_}.")
            break
        w -= eta * grad

    return w

# i: 1-D array
# n: one hot vector の次元
def one_hot_vector(i, n):
    ohvs = np.zeros((i.size, n))
    temp = np.arange(i.size)
    ohvs[temp, i[temp]] = 1
    return ohvs

def sigmoid(x):
    # np.whereは条件分岐しているのではなく2つの計算結果を合成しているだけなのでオーバーフロー警告は出る。
    # 正負の条件分岐を使ってもただ単にアンダーフローを無視しているor非正規化数を使っている（非正規化数を使うような計算は結果的に桁落ちするのでは）だけなのでオーバーフローしても問題ないと思われる。
    # return np.where(x >= 0, 1/(1+np.exp(-x)), 1-1/(1+np.exp(x)))
    return 1/(1+np.exp(-x))


def logistic_SGD(X, y, eta, eps=0, max_epochs=100000, init_w=None):
    if init_w is None:
        w = np.zeros((X.shape[1]))
    else:
        w = init_w


    rng = np.random.default_rng()
    for _ in range(max_epochs):
        i = rng.integers(0, X.shape[0])
        grad = -(y[i] - sigmoid(w @ X[i])) * X[i]
        if np.linalg.norm(grad, ord=1) < eps:
            print(f"The grad became less than eps at the epoch {_}.")
            break
        w -= eta * grad

    return w


def linear_binary_classifier(w, X):
    t = X @ w
    return np.where(t > 0, 1, 0)

def softmax(x):
    temp = np.exp(x)
    return temp/np.sum(temp)



# n: integer, 分類するクラスの数
def multi_class_logistic_SGD(X, y, n, eta, eps=0, max_epochs=100000, init_W=None, y_is_one_hot=False):
    if init_W is None:
        W = np.zeros((n, X.shape[1]))
    else:
        W = init_W

    if y_is_one_hot:
        Y_oh = y
    else:
        Y_oh = one_hot_vector(y, n)

    rng = np.random.default_rng()
    for _ in range(max_epochs):
        i = rng.integers(0, X.shape[0])
        grad = -(Y_oh[i] - softmax(W @ X[i]))[:, newaxis] @ X[i:i+1]
        if eps != 0 and np.amax(np.abs(grad)) < eps:
            print(f"The grad became less than eps at the epoch {_}.")
            break
        W -= eta * grad

    return W


def multi_linear_classifier(W, X):
    return np.argmax(W @ X.T, axis=0)

def ReLU(x):
    return np.where(x > 0, x, 0)




