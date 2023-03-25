from mylib import *
import numpy as np
from numpy import newaxis
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix

def images_to_vectors(X):
    return X.reshape(X.shape[0], -1)

def show_picture(x, y):
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(1,1,1)
    ax.set_title('Gold label: {}'.format(y))
    im = ax.imshow(x)
    fig.colorbar(im)
    plt.show()



def f(W, q, b, c, x):
    h = ReLU(W @ x + b)
    y = sigmoid(h @ q + c)
    return h, y

def my_loss(W, q, b, c, x, y):
    _, hat_y = f(W, q, b, c, x)
    return -y * np.log(hat_y) - (1-y) * np.log(1-hat_y)

def ex8_4_1():
    print("8.4 (1)")
    W = np.array([[1, 1], [-1, -1]])
    q = np.array([1,1])
    b = np.array([-0.5, 1.5])
    c = -1.5
    x = np.array([0, 0])
    h, hat_y = f(W, q, b, c, x)
    print(f"{h = :}, {hat_y = :}")
    print()

def ex8_4_2():
    print("8.4 (2)")
    x = torch.tensor([1,1], dtype=torch.float)
    W = torch.tensor([[1,1], [-1, -1]], dtype=torch.float, requires_grad=True)
    q = torch.tensor([1,1], dtype=torch.float, requires_grad=True)
    b = torch.tensor([-0.5, 1.5], dtype=torch.float, requires_grad=True)
    c = torch.tensor([-1.5], dtype=torch.float, requires_grad=True)

    loss = -(1 - (nn.ReLU()(W @ x + b) @ q + c).sigmoid()).log()
    loss.backward()
    print(f"損失関数の値: {loss.item()}")
    print(f"{W.grad = }\n{q.grad = :}\n{b.grad = }\n{c.grad = }")
    print()


def ex8_5():
    print("8.5")
    data = np.load('mnist.npz')

    X_train = images_to_vectors(data['train_x'])
    X_test = images_to_vectors(data['test_x'])
    y_train = data['train_y']
    y_test = data['test_y']

    dtype = torch.float
    X = torch.from_numpy(X_train).type(dtype)
    y = torch.from_numpy(y_train)
    X_t = torch.from_numpy(X_test).type(dtype)
    y_t = torch.from_numpy(y_test)
    numof_classes = 10


    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    class NN1(nn.Module):
        def __init__(self):
            super().__init__()
            self.lrs = nn.Sequential(
                nn.Linear(X.shape[1], numof_classes), # bias=Trueはデフォルト
            )

        def forward(self, x):
            return self.lrs(x)

    model = NN1()

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    loss_train = []
    accuracy_train = []
    loss_test = []
    accuracy_test = []

    epochs = 5
    for i in range(epochs):
        model.train()
        for Xi, yi in loader:
            loss = loss_fn(model(Xi), yi)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        loss_train.append(loss_fn(model(X), y).item())
        accuracy_train.append(torch.count_nonzero(torch.argmax(model(X), 1) == y) / len(X))
        loss_test.append(loss_fn(model(X_t), y_t).item())
        accuracy_test.append(torch.count_nonzero(torch.argmax(model(X_t), 1) == y_t) / len(X_t))


    fig, axs = plt.subplots(2, 2)
    i = np.arange(0, epochs)
    axs[0, 0].plot(i, loss_train, label="loss train")
    axs[0, 1].plot(i, accuracy_train, label="accuracy train")
    axs[1, 0].plot(i, loss_test, label="loss test")
    axs[1, 1].plot(i, accuracy_test, label="accuracy test")
    for ax in axs.flat:
        ax.set_xlabel("epoch")
        ax.legend()
    plt.show()


    pred = torch.argmax(model(X_t), 1)

    precision = np.zeros(numof_classes)
    for i in range(numof_classes):
        tempx = pred[pred == i]
        tempy = y_t[pred == i]
        precision[i] = torch.count_nonzero(tempx == tempy) / len(tempx)


    recall = np.zeros(numof_classes)
    for i in range(numof_classes):
        tempx = pred[y_t == i]
        tempy = y_t[y_t == i]
        recall[i] = torch.count_nonzero(tempx == tempy) / len(tempx)

    F1 = precision * recall / (precision + recall)

    av_precision = np.average(precision)
    av_recall = np.average(recall)
    av_F1 = np.average(F1)

    print(f"{precision = }\n{recall = }\n{F1 = }\n{av_precision = }\n{av_recall = }\n{av_F1 = }")

    print()
    print("confusion matrix")
    print(confusion_matrix(y_t, pred))
    print()

    probs = nn.Softmax(dim=1)(model(X_t)).detach().numpy()

    answer_probs = np.array([probs[i, y_t[i]] for i in range(len(y_t))])
    t = np.argsort(answer_probs)
    print("認識が簡単な事例")
    for i in t[:-4:-1]:
        print(f"テストデータインデックス: {i}, クラス: {y_t[i]}, その予測値: {answer_probs[i]}")
        show_picture(data["test_x"][i], data["test_y"][i])
    print("認識が困難な事例")
    for i in t[:3]:
        print(f"テストデータインデックス: {i}, クラス: {y_t[i]}, その予測値: {answer_probs[i]}")
        show_picture(data["test_x"][i], data["test_y"][i])

    print()

def ex8_5_5():
    print("8_5 (5)")

    data = np.load('mnist.npz')

    X_train = images_to_vectors(data['train_x'])
    X_test = images_to_vectors(data['test_x'])
    y_train = data['train_y']
    y_test = data['test_y']

    dtype = torch.float
    X = torch.from_numpy(X_train).type(dtype)
    y = torch.from_numpy(y_train)
    X_t = torch.from_numpy(X_test).type(dtype)
    y_t = torch.from_numpy(y_test)
    numof_classes = 10


    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    class NN1(nn.Module):
        def __init__(self):
            super().__init__()
            self.lrs = nn.Sequential(
                nn.Linear(X.shape[1], 30), # bias=Trueはデフォルト
                nn.ReLU(),
                nn.Linear(30, numof_classes)
            )

        def forward(self, x):
            return self.lrs(x)

    model = NN1()

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    loss_train = []
    accuracy_train = []
    loss_test = []
    accuracy_test = []

    epochs = 5
    for i in range(epochs):
        model.train()
        for Xi, yi in loader:
            loss = loss_fn(model(Xi), yi)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



        loss_train.append(loss_fn(model(X), y).item())
        accuracy_train.append(torch.count_nonzero(torch.argmax(model(X), 1) == y) / len(X))
        loss_test.append(loss_fn(model(X_t), y_t).item())
        accuracy_test.append(torch.count_nonzero(torch.argmax(model(X_t), 1) == y_t) / len(X_t))

    fig, axs = plt.subplots(2, 2)
    i = np.arange(0, epochs)
    axs[0, 0].plot(i, loss_train, label="loss train")
    axs[0, 1].plot(i, accuracy_train, label="accuracy train")
    axs[1, 0].plot(i, loss_test, label="loss test")
    axs[1, 1].plot(i, accuracy_test, label="accuracy test")
    for ax in axs.flat:
        ax.set_xlabel("epoch")
        ax.legend()
    plt.show()




if __name__ == '__main__':
    ex8_4_1()
    ex8_4_2()
    ex8_5()
    ex8_5_5()





