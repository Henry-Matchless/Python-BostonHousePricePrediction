import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


# get input data
def load_data_set(fluctuation):

    x_list = np.arange(1, 200, 2)
    np.random.shuffle(x_list)
    rnd = 2 * fluctuation * np.random.rand(len(x_list)) - fluctuation

    data_set = []
    label = []
    # get data points and target values
    for x, r in zip(x_list, rnd):
        data_set.append([x, 0.4 * x + 3 + r])
        label.append(1 if r > 0 else -1)

    data_mat = np.mat(data_set)
    data_mat_new = np.insert(data_mat, 2, values=1, axis=1)
    return data_mat_new, label


# training model
def precep_classify(data_mat, label_mat, eta=1):
    omega = np.mat(np.zeros(3))
    m = np.shape(data_mat)[0]
    error_data = True

    while error_data:
        error_data = False
        for i in range(m):
            judge = label_mat[i] * (np.dot(omega, data_mat[i].T))

            if judge <= 0:
                error_data = True
                omega = omega + eta * np.dot(label_mat[i], data_mat[i])
    return omega


# test model
def precep_test(test_data_mat, test_label_mat, omega):
    m = np.shape(test_data_mat)[0]
    error = 0.0
    for i in range(m):
        classify_num = np.dot(test_data_mat[i], omega.T)
        if classify_num > 0:
            class_ = 1
        else:
            class_ = -1
        if class_ != test_label_mat[i]:
            error += 1
    print("training error is ", float(error/m))
    return float(error/m)


# plot data
def plot(data_mat, label_mat, omega, fluctuation,error):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    X = data_mat[:, 0]
    Y = data_mat[:, 1]

    for i in range(len(label_mat)):
        if label_mat[i] > 0:
            ax.scatter(X[i].tolist(), Y[i].tolist(), color='red')
        else:
            ax.scatter(X[i].tolist(), Y[i].tolist(), color='green')
    o1 = omega[0, 0]
    o2 = omega[0, 1]
    o3 = omega[0, 2]
    x = np.linspace(0, 200, 200)
    y = (-o1 * x - o3) / o2
    ax.plot(x, y)
    plt.title("fluctuation: %s, training error: %.2f, weight: %.2f, bias: %.2f" %
              (fluctuation, error, -o1/o2, -o3/o2))
    plt.show()


# main function
def preceptron_main():
    fluctuation = 10

    # get data points and target values
    data_mat, label_mat = load_data_set(fluctuation)
    # train to get weight and bias values
    train_len = int(0.5 * len(data_mat))  # split data to training data and test data
    # train data
    omega = precep_classify(data_mat[:train_len], label_mat[:train_len])
    # test data
    error = precep_test(data_mat[train_len:], label_mat[train_len:], omega)
    # plot result
    plot(data_mat, label_mat, omega,fluctuation, error)


if __name__ == "__main__":
    preceptron_main()

