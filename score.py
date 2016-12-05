import functions as f
import validation as v
import numpy as np
from sklearn.svm import SVC
import os


def trainModel():
	train_data = f.readData("recommend_train_1hot_coding.csv")
	X = train_data[:,0:-1]
	y = train_data[:,-1]
	clf = SVC(C = 137, class_weight = 'balanced',probability = True)
	clf.fit(X,y)
	return clf



def main():
	model = trainModel()
	cur = os.getcwd()
	files = os.listdir(cur+'/recommend')
	for doc in files:
		test_data = f.readData("recommend/"+doc)
		test_data = test_data[:,0:-1]
		test_score = model.predict_log_proba(test_data)
		np.savetxt(doc[:-4] + "_score.csv", test_score, delimiter=",")


if __name__ == '__main__':
	main()