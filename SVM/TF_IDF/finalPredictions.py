import nltk
import string
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn import svm
import warnings
warnings.filterwarnings("ignore")

noOfFeatures = 100
kernelType = 'linear'#linear of rbf


x_train = []
x_validate = []
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

print 'Loading Data...'
count = 0 
totalRecords = 88639
noOfTrainingRecords = 70000
noOfValidationRecords = totalRecords-noOfTrainingRecords
with open("formatted_train_in.csv") as f:
    lis=[line.split(',') for line in f]        # create a list of lists
    for i,x in enumerate(lis):              #print the list items 
        no_punctuation = x[1].translate(None, string.punctuation)
        if(count<noOfTrainingRecords):
            x_train.append(no_punctuation)
        else:
            x_validate.append(no_punctuation)
        count+=1
        
yTrainDF = pd.DataFrame
yTrainDF = pd.read_csv("train_out_categorized1.csv",  sep=",",usecols=[1])
y_train = yTrainDF.ix[0:noOfTrainingRecords-1]
y_validate = yTrainDF.ix[noOfTrainingRecords:totalRecords-1]
print 'Done'
print 'Converting to TF-IDF Feature Set...'
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english',max_features = noOfFeatures)
tfs_train = tfidf.fit_transform(x_train)
tfs_validate = tfidf.transform(x_validate)
#print tfs_train.shape,y_train.shape
print 'Done'
print 'Training on SVM with',noOfFeatures,'features and',kernelType ,'kernel'
l_rbf_ovo = svm.SVC(kernel=kernelType, random_state=13)
l_rbf_ovo.fit(tfs_train,y_train)
print 'Trained!'
print 'Predicting on the validation set'
y_validate_predict = l_rbf_ovo.predict(tfs_validate)
#print y_validate_predict.shape
#print y_validate.shape
index = 0;
accuracy = 0
for i in y_validate_predict:
    z = y_validate.iloc[index][0]
    if(i == z):
        accuracy+=1
    index+=1
print 'Accuracy is ',accuracy*1.0/len(y_validate),'on validation set'
print 'Reading data from the test set'
x_test = []
with open("formatted_test_in.csv") as f:
    lis=[line.split(',') for line in f]        # create a list of lists
    for i,x in enumerate(lis):              #print the list items 
        no_punctuation = x[1].translate(None, string.punctuation)
        x_test.append(no_punctuation)
tfs_test = tfidf.transform(x_test)

count = 0 
fileName = 'answerSVM'+str(noOfFeatures)+'.csv'
print 'Predicting Values for Test Set..'
y_test_predict = l_rbf_ovo.predict(tfs_test)
result = pd.DataFrame(columns=('id', 'category'))
for i in y_test_predict:
    #print i
    if(i==1):
        #print 'math'
        category = 'math'
    elif(i==2):
        #print 'physics'
        category = 'physics'
    elif(i==3):
        #print 'stat'
        category= 'stat'
    elif(i==4):
        #print 'cs'
        category = 'cs'
    #x = np.array(i)b
    y= "%d"%count
    #y = count.astype(np.int32)
    #print category
    result.loc[count]=[y,category]
    #print result.loc[count]
    count+=1
result.to_csv(fileName, index=False)
print 'Prediction Completed. File Created:',fileName
