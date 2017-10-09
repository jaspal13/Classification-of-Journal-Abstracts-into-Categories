from sklearn import svm
import pandas
import numpy
import loadXDisc as lx
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

noOfFeatures = 10
kernelType = 'linear' #linear or rbf

def loadData(noOfFeatures):
	print 'Loading Data...'
	df = pandas.DataFrame(columns=('id', 'category'))
	nFeatures = noOfFeatures
	Xtrain, Xtest = lx.loadXDisc(nFeatures)
	df = pandas.read_csv('train_out_categorized1.csv')
	yTrain = df.as_matrix()
	yTrainOnlyCategories = yTrain[:,1:2]
	print 'Done'
	return Xtrain,Xtest,yTrainOnlyCategories


def splitData(Xtrain,yTrainOnlyCategories):
	print 'Spliting Data...'
	X_train_split,X_validation_split,y_train_split,y_validation_split = train_test_split(Xtrain,yTrainOnlyCategories,test_size=0.21)
	print 'Done'
	return X_train_split,X_validation_split,y_train_split,y_validation_split

def trainAndPredict(X_train_split,y_train_split,X_validation_split,kernelType):
	print 'Training Classifier..'
	classifier = svm.SVC(random_state=13,kernel=kernelType)
	classifier.fit(X_train_split,y_train_split)
	print 'Predicting For Validation Set..'
	y_validation_predict = classifier.predict(X_validation_split)
	print 'Done'
	return classifier,y_validation_predict

def measureValidationAccuracy(y_validation_predict,y_validation_split):
    print 'Calculating how well you did on the validation set'
    count = 0
    index = 0
    for i in y_validation_predict:
        if i==y_validation_split[index]:
            count += 1
        index+=1
    print count
    print "Accuracy is",(count*1.0/len(y_validation_predict))*100,"percent"
    print(metrics.classification_report(y_validation_split, y_validation_predict))
    print "1->Math,2->Physics,3->Stat,4->CS"

def getPredictionFile(classifier,Xtest,noOfFeatures):
	fileName = 'prectionSVM'+str(noOfFeatures)+'.csv'
	print 'Predicting for the test set'
	y_test_predict = classifier.predict(Xtest)
	print 'Done'
	print 'Generating prediction File'
	result = pandas.DataFrame(columns=('id', 'category'))
	count = 0
	for i in y_test_predict:
		if i=='1' or i==1:
			category = 'math'
		elif i=='2' or i==2:
			category = 'physics'
		elif i=='3'or i==3:
			category= 'stat'
		elif i=='4' or i==4:
			category = 'cs'
		y= "%d"%count
		count+=1
		if count == 5:
			break
		result.loc[count]=[y,category]
	result.to_csv('BS.csv', index=False)

Xtrain,Xtest,yTrainOnlyCategories = loadData(noOfFeatures)
X_train_split,X_validation_split,y_train_split,y_validation_split = splitData(Xtrain,yTrainOnlyCategories)
classifier,y_validation_predict = trainAndPredict (X_train_split,y_train_split,X_validation_split,kernelType)
measureValidationAccuracy(y_validation_predict,y_validation_split)
getPredictionFile(classifier,Xtest,noOfFeatures)



