from mylib import *
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
import scipy




if __name__ == '__main__':
    plt.rcParams["font.family"] = "MS Gothic"
    data = np.loadtxt("SSDSE-C-2022.csv", dtype=np.str_, delimiter=",")
    kenchou_names = data[3:, 2]
    feature_vectors = data[3:, 3:].astype(np.float_)

    Z = scipy.cluster.hierarchy.linkage(feature_vectors, 'single')

    fig, ax = plt.subplots()
    dn = scipy.cluster.hierarchy.dendrogram(Z, labels=kenchou_names, ax=ax)
    ax.set_ylabel("Distance")
    ax.set_title("最短距離法")
    plt.show()


    Z = scipy.cluster.hierarchy.linkage(feature_vectors, 'complete')

    fig, ax = plt.subplots()
    dn = scipy.cluster.hierarchy.dendrogram(Z, labels=kenchou_names, ax=ax)
    ax.set_ylabel("Distance")
    ax.set_title("最長距離法")
    plt.show()

    Z = scipy.cluster.hierarchy.linkage(feature_vectors, 'ward')

    fig, ax = plt.subplots()
    dn = scipy.cluster.hierarchy.dendrogram(Z, labels=kenchou_names, ax=ax)
    ax.set_ylabel("Distance")
    ax.set_title("Ward法")
    plt.show()
