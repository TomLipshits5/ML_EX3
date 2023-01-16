import math

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


class Regressor:
    def __init__(self, w: np.array):
        self.w = w

    def predict(self, x: np.array):
        if x.shape == self.w.shape:
            return np.dot(self.w, x)
        raise Exception("shape error for dot product")


def getRidgeRegressor(X: np.array, Y: np.array, l: int):
    d, m = X.shape
    X_t = np.transpose(X)
    I_m = np.eye(d)
    invert = np.linalg.inv((X @ X_t) + (l * I_m))
    Xy = X @ Y
    return Regressor(np.squeeze(invert @ Xy))


def getDataSets(fileNAme: str):
    data = sio.loadmat(fileNAme)
    return (data['X']), (data['Y']), (data['Xtest']), (data['Ytest'])


def getMeanSquaredError(reg: Regressor, testX: np.array, testY: np.array):
    d, m = testX.shape
    sum = 0.0
    for i in range(testX.shape[1]):
        x = testX[:, i]
        prediction = reg.predict(x.reshape(reg.w.shape))
        sum += math.pow((prediction - testY[i]), 2)
    return sum / m


def evalAllLambdas(trainX: np.array, trainY: np.array, testX: np.array, testY: np.array):
    results = []
    for l in range(31):
        reg = getRidgeRegressor(trainX, trainY, l)
        results.append(getMeanSquaredError(reg, testX, testY))
    return results


def evalAllTrainSizes(trainX: np.array, trainY: np.array, testX: np.array, testY: np.array):
    trainSizes = list(range(10, 110, 10))
    results = {}
    for trainSize in trainSizes:
        res = evalAllLambdas(trainX[:, :trainSize], trainY[:trainSize:], testX, testY)
        results[trainSize] = res
    return results


def task2a(results: dict):

    points = {}
    for trainSize, results in results.items():

        l = results.index(min(results))
        points[trainSize] = l


    plt.yticks(list(range(0, 31, 5)))
    plt.xticks(list(range(0,120,10)))
    plt.xlabel("Sample-Size")
    plt.ylabel("Lambda")
    plt.title("Best lambda value for sample size")
    plt.scatter(points.keys(), points.values(), c=np.random.rand(10), alpha=0.5)
    plt.show()



if __name__ == '__main__':
    print("starting:")
    fileName = 'regdata.mat'
    trainX, trainY, testX, testY = getDataSets(fileName)
    results = evalAllTrainSizes(trainX, trainY, testX, testY)
    task2a(results)
