import functions as f
import validation as v
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

def trainSVMWithGaussianKernel(data, train, validation, C):
	'''
	SVM with Gaussian Kernal
	'''
	w = open("F-1 for Support Vector Mahines.txt", "a+")
	n = len(C)
	f1 = np.zeros(n)
	X_train, y_train = v.makeMatrix(data, train)
	X_test, y_test = v.makeMatrix(data, validation)

	for i in range(n):
		clf = SVC(C = C[i], class_weight = 'balanced')
		f1[i] = v.validate(data, X_train, y_train, X_test, y_test, clf)
		print("F-1 measure for SVM with RBF and C = %s is %s" 
			%(C[i], f1[i]))

	f1_max = np.max(f1)
	optimal_C = C[np.argmax(f1)]
	print("Highest F-1 measure for SVM with RBF is %s with C = %s" %(f1_max, optimal_C))
	w.write("Highest F-1 measure for SVM with RBF is %s with C = %s\n" 
		%(f1_max, optimal_C))
	w.close()

	return f1_max, optimal_C


def trainSVMWithLinearKernel(data, train, validation, C, 
	loss = 'squared_hinge', penalty = 'l2'):
	'''
	SVM with Linear Kernal with LinearSVC()
	'''
	w = open("F-1 for Support Vector Mahines.txt", "a+")
	n = len(C)
	f1 = np.zeros(n)
	X_train, y_train = v.makeMatrix(data, train)
	X_test, y_test = v.makeMatrix(data, validation)

	for i in range(n):
		clf = LinearSVC(C = C[i], loss = loss, penalty = penalty, 
			class_weight = 'balanced', dual = True)
		f1[i] = v.validate(data, X_train, y_train, X_test, y_test, clf)
		print("F-1 measure for SVM with Linear Kernal, %s loss, %s penalty, and C = %s is %s" %(loss, penalty, C[i], f1[i]))

	f1_max = np.max(f1)
	optimal_C = C[np.argmax(f1)]
	print("Highest F-1 measure for SVM with Linear Kernal, %s loss and %s penalty is %s with C = %s" %(loss, penalty, f1_max, optimal_C))
	w.write("Highest F-1 measure for SVM with Linear Kernal, %s loss and %s penalty is %s with C = %s\n" %(loss, penalty, f1_max, optimal_C))
	w.close()

	return f1_max, optimal_C


def trainSVMWithLinearKernel2(data, train, validation, C):
	'''
	SVM with Linear Kernal with SVC()
	'''
	w = open("F-1 for Support Vector Mahines.txt", "a+")
	n = len(C)
	f1 = np.zeros(n)
	X_train, y_train = v.makeMatrix(data, train)
	X_test, y_test = v.makeMatrix(data, validation)

	for i in range(n):
		clf = SVC(C = C[i], kernel = 'linear', class_weight = 'balanced')
		f1[i] = v.validate(data, X_train, y_train, X_test, y_test, clf)
		print("F-1 measure for SVM with Linear kernel and C = %s is %s" %(C[i], f1[i]))

	f1_max = np.max(f1)
	optimal_C = C[np.argmax(f1)]
	print("Highest F-1 measure for SVM with Linear is %s with C = %s" %(f1_max, optimal_C))
	w.write("Highest F-1 measure for SVM with Linear is %s with C = %s\n" %(f1_max, optimal_C))
	w.close()

	return f1_max, optimal_C


def main():
	np.random.seed(0)

	data = f.readData()
	train, validation, test = f.splitData(data.shape[0])

	# C = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
	C = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005]
	# C = [10, 50, 100, 500, 1000]
	# C = np.arange(1, 10, 1)

	# C = 137, F-1 = 0.541310541311
	trainSVMWithGaussianKernel(data, train, validation, C)
	
	trainSVMWithLinearKernel(data, train, validation, C)
	
	# SVM with Linear Kernel
 
	# l1, squared hinge, C = 50, F-1 = 0.525447042641
	# l2, hinge, C = 0.001 , F-1 = 0.512968299712
	# l2, squared hinge, C = 1, F-1 = 0.524725274725

	trainSVMWithLinearKernel2(data, train, validation, C)
	
	

if __name__ == '__main__':
	main()