from mylib import *
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans




if __name__ == '__main__':
    data = np.loadtxt("SSDSE-C-2022.csv", dtype=np.str_, delimiter=",")
    kenchou_names = data[3:, 2]
    feature_vectors = data[3:, 3:].astype(np.float_)

    K = 5
    K_max = 15
    inertias = []
    for K in range(1, K_max+1):
        model = KMeans(n_clusters=K)
        model.fit(feature_vectors)
        inertias.append(model.inertia_)

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, K_max+1), inertias, "o-")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel("inertia")
    plt.show()


    K = 4
    print(f"{K = }")
    model = KMeans(n_clusters=K)
    model.fit(feature_vectors)
    for i in range(K):
        print("クラスタ", i+1)
        print(kenchou_names[model.labels_ == i])








