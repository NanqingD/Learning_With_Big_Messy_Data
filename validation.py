import numpy as np
import math
from functools import reduce


def partitionForCrossValidation(train, k = 5):
    n = len(train)
    p = math.floor(n / k) + 1
    partitioned = []
    for i in range(k):
        if i < k - 1:
            partitioned.append(train[i*p:(i+1)*p])
        else:
            partitioned.append(train[i*p:])
    
    print("Data partitioned.")
    return np.array(partitioned)


def leaveOneOut(i, partitioned, k):
    _index = list(range(k))
    _test = partitioned[i]
    _index.pop(i)
    _train = list(map(lambda x: partitioned[x], _index))
    _train = reduce((lambda x,y:np.append(x,y)), _train)
    return _train,_test


def kFoldCrossValidation(data, train, classifier, k = 5):
    f1_score = np.zeros(k)
    print("Cross-Validation started.")
    partitioned = partitionForCrossValidation(train, k)

    for i in range(k):
        _train, _test = leaveOneOut(i, partitioned, k) 
        f1_score[i] = naiveValidation(data, _train, _test, classifier)
    
    mean_f1_score = np.mean(f1_score)
    print("F-1 measure is %s" %(mean_f1_score))
    return mean_f1_score


def makeMatrix(data, index_):
    X = data[index_,0:-1]
    y = data[index_,-1]
    return X, y


def validate(data, X_train, y_train, X_test, y_test, classifier):
    classifier.fit(X_train, y_train)
    # print("Model fitted.")
    y_pred = classifier.predict(X_test)
    # print("Prediction finished.")
    f1_score = calculateF1Score(y_pred, y_test)
    # print("F-1 measure is %s" %(f1_score))
    return f1_score


def calculateF1Score(y_pred, y_test):
    TP = np.sum((y_pred == 1) & (y_test == 1))
    FP = np.sum((y_pred == 1) & (y_test == 0))
    FN = np.sum((y_pred == 0) & (y_test == 1))
    prec = TP / (TP + FN)
    recall = TP / (TP + FP)
    return 2* prec * recall / (prec + recall)


def bootstrap(train, size):
    return np.random.choice(train, size, replace=False)


def overSampling(train, positive, negative, size):
    _positive = np.random.choice(positive, math.ceil(size/2), replace=True)
    _negative = np.random.choice(negative, math.ceil(size/2), replace=False)
    return np.append(_positive, _negative)


def underSampling(train, positive, negative, size = 4000):
    _positive = np.random.choice(positive, math.ceil(size/2), replace=False)
    _negative = np.random.choice(negative, math.ceil(size/2), replace=False)
    return np.append(_positive, _negative)


def bootstrapValidation(data, train, validation, classifier, sampler = bootstrap, 
    size = 10000, positive = None, negative = None, n = 1):
    """
    sampler: bootstrap, overSampling, underSampling 
    """
    f1_score = np.zeros(n)
    print("Validation started.")
    for i in range(n):
        _train = []
        if sampler == bootstrap:
            _train = sampler(train,size)
        elif sampler == underSampling:
            _train = sampler(train, positive, negative)
        elif sampler == overSampling:
            _train = sampler(train, positive, negative, size)

        f1_score[i] = Validation(data, _train, validation, classifier)

    mean_f1_score = np.mean(f1_score)
    print("F-1 measure is %s" %(mean_f1_score))
    return mean_f1_score


def getPositiveIndex(data, train):
    positive = set(np.where(data[:,-1] == 1)[0])
    positive = list(positive & set(train))
    return positive


def getNegativeIndex(data, train):
    negative = set(np.where(data[:,-1] == 0)[0])
    negative = list(negative & set(train))
    return negative


def reportTestF1Score(data, train, test, classifier):
    print("Validation on Test")
    f1 = Validation(data, train, validation, classifier)
    print("F-1 measure is %s" %(f1))
    return f1