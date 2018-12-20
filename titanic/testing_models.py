# Importing important libraries
import csv
import numpy as np
import statistics as stats
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
#Functions of different algorithms

# Logistic_regression
def log_reg(Xtrain, ytrain, Xval, yval):

	# Training on Data
	reg = linear_model.LogisticRegression()
	reg.fit(Xtrain, ytrain)

	# Predicting
	ypred = reg.predict(Xval)

	#calculating score
	return [reg.score(Xtrain,ytrain), reg.score(Xval,yval)]

# SVM
def s_v_m(Xtrain, ytrain, Xval, yval):

	# Training on Data
	clf = SVC()
	clf.fit(Xtrain, ytrain)

	# Predicting
	ypred = clf.predict(Xval)

	#calculating score
	return [clf.score(Xtrain ,ytrain), clf.score(Xval,yval)]

# Neural Network
def n_n(Xtrain, ytrain, Xval, yval):

	# Training on Data
	nn = MLPClassifier(hidden_layer_sizes=(15,10))
	nn.fit(Xtrain, ytrain)

	# Predicting
	ypred = nn.predict(Xval)

	#calculating score
	return [nn.score(Xtrain,ytrain), nn.score(Xval,yval)]

# Random forest
def rf(Xtrain, ytrain, Xval, yval):

	# Training on Data
	nn = RandomForestClassifier(n_estimators = 50)
	nn.fit(Xtrain, ytrain)

	# Predicting
	ypred = nn.predict(Xval)

	#calculating score
	return [nn.score(Xtrain,ytrain), nn.score(Xval,yval)]

#function used to do feature scaling
def feature_scaling(list):
	mean = stats.mean(list)
	var = (stats.stdev(list))**2
	return map(lambda x: ((x-mean)/var), list)

# Declearing lists to store data
survived = []
pclass = []
sex = []
age = []
sibSp = []
parch = []
fare = []
embarked = []

# Reading csv File
with open("train.csv") as csvfile:
	readCsv = csv.reader(csvfile, delimiter=',')
	# Storing Data
	next(readCsv)
	for rows in readCsv:
		survived.append(float(rows[1]))
		pclass.append(float(rows[2]))
		if rows[4] == 'male':
			sex.append(1.0)
		else:
			sex.append(0.0)
		if rows[5] == '':
			age.append(0.0)
		else:
			age.append(float(rows[5]))
		sibSp.append(float(rows[6]))
		parch.append(float(rows[7]))
		fare.append(float(rows[9]))
		if rows[11] == 'S':
			embarked.append(1.0)
		elif rows[11] == 'C':
			embarked.append(2.0)
		else:
			embarked.append(3.0)

# Transforimg lists by appling feature scaling 
pclass = list(feature_scaling(pclass))
sex = list(feature_scaling(sex))
age = list(feature_scaling(age))
sibSp = list(feature_scaling(sibSp))
parch = list(feature_scaling(parch))
fare = list(feature_scaling(fare))
embarked = list(feature_scaling(embarked))

# Declaring necessary Variables
n = len(survived)
trainDataLen = int(0.8*n)

#Declaring X and y
Xtrain = np.transpose(np.array([pclass[0: trainDataLen], sex[0: trainDataLen], age[0: trainDataLen], sibSp[0: trainDataLen], parch[0: trainDataLen], fare[0: trainDataLen], embarked[0: trainDataLen]]))
ytrain = np.array(survived[0: trainDataLen])
Xval = np.transpose(np.array([pclass[trainDataLen:], sex[trainDataLen:], age[trainDataLen:], sibSp[trainDataLen:], parch[trainDataLen:], fare[trainDataLen:], embarked[trainDataLen:]]))
yval = np.array(survived[trainDataLen:])

# Logistic regression
print ("score on logistic regression = %s" %log_reg(Xtrain, ytrain, Xval, yval))

# SVM
print ("score on SVM = %s" %s_v_m(Xtrain, ytrain, Xval, yval))

# Neural Network

print ("score on Neural Network = %s" %n_n(Xtrain, ytrain, Xval, yval))

# random forest
print ("score on random forest = %s" %rf(Xtrain, ytrain, Xval, yval))

#map filter reduce
