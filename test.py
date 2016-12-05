import functions as f
import validation as v
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


def main():
	np.random.seed(0)

	data = f.readData()
	train, validation, test = f.splitData(data.shape[0])

	X_train, y_train = v.makeMatrix(data, train)
	X_test, y_test = v.makeMatrix(data, test)

	print("Logistic Regression")
	clf = LogisticRegression(C = 0.061, class_weight = 'balanced', max_iter = 10000, solver = 'sag', n_jobs = -1)
	f1 = v.validate(data, X_train, y_train, X_test, y_test, clf)
	print("F-1 measure for Logistic Regression with l2 and C = %s is %s" %(0.061, f1))

	clf = LogisticRegression(penalty = 'l1', C = 0.175, class_weight = 'balanced', max_iter = 5000, n_jobs = -1)
	f1 = v.validate(data, X_train, y_train, X_test, y_test, clf)
	print("F-1 measure for Logistic Regression with l1 and C = %s is %s" %(0.175, f1))

	print("SVM")
	clf = SVC(C = 137, class_weight = 'balanced')
	f1 = v.validate(data, X_train, y_train, X_test, y_test, clf)
	print("F-1 measure for SVM with RBF and C = %s is %s" %(137, f1))


	# l1, squared hinge, C = 50, F-1 = 0.525447042641
	# l2, hinge, C = 0.001 , F-1 = 0.512968299712
	# l2, squared hinge, C = 1, F-1 = 0.524725274725
	C = 50
	loss = "squared_hinge"
	penalty = 'l1'
	clf = LinearSVC(C = C, loss = loss, penalty = penalty, class_weight = 'balanced', dual = False)
	f1 = v.validate(data, X_train, y_train, X_test, y_test, clf)
	print("F-1 measure for SVM with Linear Kernal, %s loss, %s penalty, and C = %s is %s" %(loss, penalty, C, f1))

	C = 0.001
	loss = "hinge"
	penalty = 'l2'
	clf = LinearSVC(C = C, loss = loss, penalty = penalty, class_weight = 'balanced', dual = True)
	f1 = v.validate(data, X_train, y_train, X_test, y_test, clf)
	print("F-1 measure for SVM with Linear Kernal, %s loss, %s penalty, and C = %s is %s" %(loss, penalty, C, f1))

	C = 1
	loss = "squared_hinge"
	penalty = 'l2'
	clf = LinearSVC(C = C, loss = loss, penalty = penalty, class_weight = 'balanced', dual = False)
	f1 = v.validate(data, X_train, y_train, X_test, y_test, clf)
	print("F-1 measure for SVM with Linear Kernal, %s loss, %s penalty, and C = %s is %s" %(loss, penalty, C, f1))

	# max_features = None
	# clf = RandomForestClassifier(n_estimators = 100, max_features = max_features, n_jobs = -1, class_weight = 'balanced')
	# f1 = v.validate(data, X_train, y_train, X_test, y_test, clf)
	# print("F-1 measure for Random Forest with %s trees and max_features = %s is %s" %(100, max_features, f1))


	# clf = GradientBoostingClassifier(n_estimators = N[i], max_features = max_features)
	# f1[i] = v.validate(data, X_train, y_train, X_test, y_test, clf)
	# print("F-1 measure for Gradient Boosting with %s boosting stages and max_features = %s is %s" %(N[i], max_features, f1))

	# N = 100
	# alpha = 0.001
	# clf = MLPClassifier(hidden_layer_sizes = N, activation = 'logistic', alpha = alpha, learning_rate = 'adaptive', max_iter  = 10000)
	# f1 = v.validate(data, X_train, y_train, X_test, y_test, clf)
	# print("F-1 measure for Neural Network with %s hidden layers, %s activation and alpha = %s is %s"  %(N, activation, alpha, f1))








if __name__ == '__main__':
	main()
