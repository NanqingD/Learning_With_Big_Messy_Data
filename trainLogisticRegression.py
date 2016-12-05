#!/usr/bin/env python

import functions as f
import validation as v
import numpy as np
from sklearn.linear_model import LogisticRegression


def trainRegularizationStrengthForl2(data, train, validation, C):
	'''
	Logistic Regression with Ridge
	'''
	w = open("F-1 for Logistic Regression.txt", "a+")
	n = len(C)
	f1 = np.zeros(n)
	X_train, y_train = v.makeMatrix(data, train)
	X_test, y_test = v.makeMatrix(data, validation)

	for i in range(n):
		clf = LogisticRegression(C = C[i], class_weight = 'balanced', 
			max_iter = 10000, solver = 'sag', n_jobs = -1)
		f1[i] = v.validate(data, X_train, y_train, X_test, y_test, clf)
		print("F-1 measure for Logistic Regression with l2 and C = %s is %s" 
			%(C[i], f1[i]))

	f1_max = np.max(f1)
	optimal_C = C[np.argmax(f1)]
	print("Highest F-1 measure for Logistic Regression with l2 is %s with C = %s" 
		%(f1_max, optimal_C))
	w.write("Highest F-1 measure for Logistic Regression with l2 is %s with C = %s\n" 
		%(f1_max, optimal_C))
	w.close()

	return f1_max, optimal_C


def trainRegularizationStrengthForl1(data, train, validation, C):
	'''
	Logistic Regression with LASSO
	'''
	w = open("F-1 for Logistic Regression.txt", "a+")
	n = len(C)
	f1 = np.zeros(n)
	X_train, y_train = v.makeMatrix(data, train)
	X_test, y_test = v.makeMatrix(data, validation)

	for i in range(n):
		clf = LogisticRegression(penalty = 'l1', C = C[i], class_weight = 'balanced', 
			max_iter = 5000, n_jobs = -1)
		f1[i] = v.validate(data, X_train, y_train, X_test, y_test, clf)
		print("F-1 measure for Logistic Regression with l1 with C = %s is %s" 
			%(C[i], f1[i]))

	f1_max = np.max(f1)
	optimal_C = C[np.argmax(f1)]
	print("Highest F-1 measure for Logistic Regression with l1 is %s with C = %s" 
		%(f1_max, optimal_C))
	w.write("Highest F-1 measure for Logistic Regression with l1 is %s with C = %s\n" 
		%(f1_max, optimal_C))
	w.close()

	return f1_max, optimal_C


def main():
	np.random.seed(0)

	data = f.readData()
	train, validation, test = f.splitData(data.shape[0])

	C = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

	# C = 0.061 F-1 = 0.525
	trainRegularizationStrengthForl2(data, train, validation, 
		C)
	
	# C = 0.175 F-1 = 0.526170798898
	trainRegularizationStrengthForl1(data, train, validation, 
		C)


if __name__ == '__main__':
	main()