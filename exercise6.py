from mylib import *
import numpy as np
from numpy import newaxis


# データの最後には定数項1が付加されている。
def images_to_vectors(X):
    X = np.reshape(X, (len(X), -1))         # Flatten: (N x 28 x 28) -> (N x 784)
    return np.c_[X, np.ones(len(X))]        # Append 1: (N x 784) -> (N x 785)


if __name__ == '__main__':
    data = np.load('mnist.npz')

    X_train = images_to_vectors(data['train_x'])
    X_test = images_to_vectors(data['test_x'])
    y_train = data['train_y']
    y_test = data['test_y']

    W = multi_class_logistic_SGD(X_train, y_train, 10, 0.01, max_epochs=10000)
    y_pred = multi_linear_classifier(W, X_test)
    print("正解率:", np.count_nonzero(y_pred == y_test)/y_test.size)






