from mylib import *
import numpy as np
from numpy import newaxis
import matplotlib as mpl
import matplotlib.pyplot as plt
import collections
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer


def tokenize(s):
    return [t.rstrip('.') for t in s.split(' ')]

def vectorize(tokens):
    return collections.Counter(tokens)

def readiter(fi):
    for line in fi:
        fields = line.strip('\n').split('\t')
        x = vectorize(tokenize(fields[1]))
        y = fields[0]
        yield x, y


if __name__ == '__main__':
    with open('smsspamcollection/SMSSpamCollection', encoding="utf-8") as fi:
        D = [d for d in readiter(fi)]

    Dtrain, Dtest = train_test_split(D, test_size=0.1, random_state=0)

    VX = DictVectorizer()
    VY = LabelEncoder()

    # Xtrainはscipyのscipy.sparce.csr_matrix。ndarrayに変換するには.toarray()を使う。Ytrainはndarray
    # csr_matrix a に対してprint(a)はndarrayのような表記ではなく aの各インデックスiに対して "(i, a[i]の非ゼロ要素のインデックスj) a[i, j]の値" の形式で表示される。
    # Xtrainの行 x は、特徴（ここでは英単語）と一対一に対応付けた**インデックス**に対して、xのそのインデックスの値が特徴の値となっているベクトルである。
    Xtrain = VX.fit_transform([d[0] for d in Dtrain]).toarray()
    Ytrain = VY.fit_transform([d[1] for d in Dtrain])
    Xtest = VX.transform([d[0] for d in Dtest]).toarray()
    Ytest = VY.transform([d[1] for d in Dtest])

    eta = 0.01
    # init_w = 0 だと初期状態から正解率が高かったので変えている。勾配はかなり小さい様子
    w = logistic_SGD(Xtrain, Ytrain, eta, max_epochs=10000, init_w=np.ones((Xtrain.shape[1])))

    Y_pred = linear_binary_classifier(w, Xtest)
    print("正解率:", np.count_nonzero(Y_pred == Ytest)/Ytest.size)

    # トップ20では正のワードがなかったのでトップ50
    indices = np.flip(np.argsort(np.abs(w))[-50:])
    print("重みが正に大きいワード")
    for i in indices:
        if w[i] >= 0:
            print(VX.feature_names_[i], ":", w[i])
    print("重みが負に大きいワード")
    for i in indices:
        if w[i] < 0:
            print(VX.feature_names_[i], ":", w[i])





