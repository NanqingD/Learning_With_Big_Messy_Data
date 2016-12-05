import functions as f
import validation as v
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

def trainRandomForest(data, train, validation, N, max_features = None):
	w = open("F-1 for Random Forest.txt", "a+")
	n = len(N)
	f1 = np.zeros(n)
	X_train, y_train = v.makeMatrix(data, train)
	X_test, y_test = v.makeMatrix(data, validation)

	for i in range(n):
		clf = RandomForestClassifier(n_estimators = N[i], max_features = max_features, n_jobs = -1, class_weight = 'balanced')
		f1[i] = v.validate(data, X_train, y_train, X_test, y_test, clf)
		print("F-1 measure for Random Forest with %s trees and max_features = %s is %s" %(N[i], max_features, f1[i]))

	# f1_max = np.max(f1)
	# optimal_N = N[np.argmax(f1)]
	# print("Highest F-1 measure for Random Forest with %s trees and max_features = %s is %s" %(optimal_N, max_features, f1_max))
	# w.write("Highest F-1 measure for Random Forest with %s trees and max_features = %s is %s\n" %(optimal_N, max_features, f1_max))

	f1_mean = np.mean(f1)
	optimal_N = N[0]
	print("Average F-1 measure for Random Forest with %s trees and max_features = %s is %s" %(optimal_N, max_features, f1_mean))
	w.write("Average F-1 measure for Random Forest with %s trees and max_features = %s is %s\n" %(optimal_N, max_features, f1_mean))

	return f1_mean, optimal_N


def trainGradientBoosting(data, train, validation, N, max_features = None):
	w = open("F-1 for Gradient Boosting.txt", "a+")
	n = len(N)
	f1 = np.zeros(n)
	X_train, y_train = v.makeMatrix(data, train)
	X_test, y_test = v.makeMatrix(data, validation)

	for i in range(n):
		clf = GradientBoostingClassifier(n_estimators = N[i], max_features = max_features)
		f1[i] = v.validate(data, X_train, y_train, X_test, y_test, clf)
		print("F-1 measure for Gradient Boosting with %s boosting stages and max_features = %s is %s" %(N[i], max_features, f1[i]))

	f1_max = np.max(f1)
	optimal_N = N[np.argmax(f1)]
	print("Highest F-1 measure for Gradient Boosting with %s boosting stages and max_features = %s is %s" %(optimal_N, max_features, f1_max))
	w.write("Highest F-1 measure for Gradient Boosting with %s boosting stages and max_features = %s is %s\n" %(optimal_N, max_features, f1_max))

	# f1_mean = np.mean(f1)
	# optimal_N = N[0]
	# print("Average F-1 measure for Gradient Boosting with %s boosting stages and max_features = %s is %s" %(optimal_N, max_features, f1_mean))
	# w.write("Average F-1 measure for Gradient Boosting with %s boosting stages and max_features = %s is %s\n" %(optimal_N, max_features, f1_mean))

	return f1_max, optimal_N
	# return f1_mean, optimal_N


def main():
	np.random.seed(0)

	data = f.readData()
	train, validation, test = f.splitData(data.shape[0])

	# C = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

	# n_estimators = [5000, 10000, 50000, 100000, 500000]
	# n_estimators = np.arange(10, 200, 10)
	# n_estimators = np.repeat([100],100)

	# number_of_trees = 100, average F-1 on 100 forests = 0.377228139802
	# trainRandomForest(data, train, validation, n_estimators, max_features = None)

	n_estimators = [100, 200, 500, 1000]
	# number_of_boosting_stages = 100, average F-1 on 100 boostings = 0.377228139802
	trainGradientBoosting(data, train, validation, n_estimators, max_features = 'auto')

# F-1 measure for Gradient Boosting with 5000 boosting stages and max_features = None is 0.299435028249
# F-1 measure for Gradient Boosting with 10000 boosting stages and max_features = None is 0.310249307479
# F-1 measure for Gradient Boosting with 50000 boosting stages and max_features = None is 0.327956989247
# F-1 measure for Gradient Boosting with 100000 boosting stages and max_features = None is 0.337801608579
if __name__ == '__main__':
	main()