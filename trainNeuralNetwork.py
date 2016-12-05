import functions as f
import validation as v
import numpy as np
from sklearn.neural_network import MLPClassifier


def trainNeuralNetworks(data, train, validation, N = 50, activation = 'logistic', alpha = 0.001, rep = 10):
	# w = open("F-1 for Neural Network.txt", "a+")
	X_train, y_train = v.makeMatrix(data, train)
	X_test, y_test = v.makeMatrix(data, validation)
	f1 = np.zeros(rep)
	for i in range(rep):
		clf = MLPClassifier(hidden_layer_sizes = N, activation = 'logistic', alpha = alpha, learning_rate = 'adaptive', max_iter  = 10000)
		f1[i] = v.validate(data, X_train, y_train, X_test, y_test, clf)
	
	f1_mean = np.mean(f1)
	print("Average F-1 measure for Neural Network with %s hidden layers, %s activation and alpha = %s is %s"  %(N, activation, alpha, f1_mean))
	return f1_mean
	# w.close()


def main():
	np.random.seed(0)

	data = f.readData()
	train, validation, test = f.splitData(data.shape[0])

	# trainNeuralNetworks(data, train, validation)


	alphas = np.arange(0.0001,0.0015, 0.0001) #[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
	N = [75, 100]
	F_1 = np.zeros([len(alphas), len(N)])
	for i in range(len(alphas)):
		for j in range(len(N)):
			F_1[i, j] = trainNeuralNetworks(data, train, validation, N = N[j], alpha = alphas[i])

	# Average F-1 measure for Neural Network with 100 hidden layers, logistic activation, learning_rate = 'adaptive'and alpha = 0.001 is 0.2584181792





if __name__ == '__main__':
	main()