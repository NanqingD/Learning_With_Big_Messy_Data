import functions as f
import validation as v
import numpy as np
import math
from sklearn.svm import SVC


def splitData(n):
	np.random.seed(0)
	split = math.floor(n * 0.9)
	permutated_index = np.random.permutation(np.arange(n))
	train = permutated_index[:split]
	test = permutated_index[split:]
	print("Data splitting finished.")
	return train, test


def resampleTrain(train):
	np.random.seed()
	n = len(train)
	train = np.random.choice(train, math.floor(n*8/9) ,replace=True)
	return train



n = 1000
f1 = np.zeros(n)
clf = SVC(C = 137, class_weight = 'balanced')
data = f.readData()
train, test = splitData(data.shape[0])
for i in range(n):
	print(i+1)
	train_ = resampleTrain(train)

	X_train, y_train = v.makeMatrix(data, train_)
	X_test, y_test = v.makeMatrix(data, test)
	try:
		f1[i] = v.validate(data, X_train, y_train, X_test, y_test, clf)
	except Exception:
		f1[i] = 0



np.savetxt("f1_confidence.csv", f1, delimiter=",")
print(np.mean(f1))
print(np.std(f1))


