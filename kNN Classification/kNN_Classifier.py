import re
import sys
import nltk
import math
import numpy as np
import pandas as pd
import pprint, pickle
from bs4 import BeautifulSoup
from operator import itemgetter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# nltk.download()  # Download text data sets, including stop words
from nltk.corpus import stopwords  # Import the stop word list
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer


# =========================================================================
#                   Function to load the training data.
#   (Ensure that the directory structure is the same when running the code)
# =========================================================================
def loadTrainIn():
    return pd.read_csv("../datasets/train_in.csv", header=0, sep=",")

# =========================================================================
#                   Function to load the test data.
#   (Ensure that the directory structure is the same when running the code)
# =========================================================================
def loadTestIn():
    return pd.read_csv("../datasets/test_in.csv", header=0, sep=",")

# =========================================================================
#    Function to clean the given abstract. Takes an abstract as input.
#       Does the following:
#            * Removing punctuation and symbols
#            * Removing "stop words" according to NLTK package
#            * Stemming the words using SnowballStemmer
#            * Lemmatizing the stemmed words
#   The output is a single string (a cleaned abstract)
# =========================================================================
def cleanAbstract(raw_abstract):
    # 1. Remove HTML (commneted out since its not necessary)
    # abstract_text = BeautifulSoup(raw_abstract, "lxml").get_text()

    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", raw_abstract)

    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()

    # 4. Convert the stop words to a set
    stops = set(stopwords.words("english"))

    # 5. Remove stop words using the stop words list in the NLTK package
    meaningful_words = [w for w in words if not w in stops]

    # 6. Stem and lemmatize the words and remove repeating words
    stemmer = SnowballStemmer("english")
    lemmatizer = WordNetLemmatizer()

    for index in range(0, len(meaningful_words)):
        meaningful_words[index] = stemmer.stem(meaningful_words[index])
        meaningful_words[index] = lemmatizer.lemmatize(meaningful_words[index])

    meaningful_words = list(set(meaningful_words))

    # 7. Join the words back into one string separated by space, and return the result.
    return (" ".join(meaningful_words))


# =========================================================================
#                   Function to clean the given data set.
#               The input is a list of abstracts to be cleaned.
#         The output is a pandas dataframe with cleaned abstracts
# =========================================================================
def cleanSet(setToClean, start, end):
    print("Cleaning and parsing the given set journal abstracts...\n")
    formatted_Set = []

    for index in range(start, end):
        # for index in range(0, 5):
        formatted_abstract = []
        rich_abstract = cleanAbstract(setToClean['abstract'][index])

        # If the index is evenly divisible by 10000, print a message
        if ((index + 1) % 10000 == 0):
            print("Abstract %d of %d\n" % (index + 1, end))

        formatted_abstract.append(setToClean['id'][index])
        formatted_abstract.append(rich_abstract)
        formatted_Set.append(formatted_abstract)

    df = pd.DataFrame(formatted_Set)
    return df


# =========================================================================
#   Function to find the number of times a specific label/category
#                                               occurs in the given list.
#        Input - category, list of abstracts with categories
#        Output - total count of the specific category in the given list.
# =========================================================================
def findLabelCount(label, listOfItems):
    totalCount = 0
    for item in listOfItems:
        if item[2] == label:
            totalCount = totalCount + 1
    return totalCount


# =========================================================================
#   Function to find the total sum of scores of a specific label/category
#                                                       in the given list.
#        Input - category, list of abstract scores with categories
#        Output - total score of the given category.
# =========================================================================
def getTotalScoreOfLabel(label, listOfScores):
    totalScore = 0
    for item in listOfScores:
        if item[2] == label:
            totalScore = totalScore + item[1]
    return totalScore

# =========================================================================================================
#       Method that returns the class/label of a given new "Abstract" text
#         from a research journal.
#                          k: the number of nearest neighbours to be considered
#                newPhraseId: the id of the new phrase whose class is to be determined
#                  newPhrase: the new phrase whose class is to be determined
#            formatted_train: the list of cleaned and formatted training samples
#          train_results_set: the category list of the training samples with their id's
#                 vectorizer: the CountVectorized object with the feature-word list and their frequencies
#        X_train_tfidf_array: the tf-idf value array of each feature-word in every training sample.
# =========================================================================================================
def findBestLabel(k, newPhraseId, newPhrase, formatted_train, train_results_set, vectorizer, X_train_tfidf_array):
    global label_math, label_cs, label_physics, label_stat

    scoresPerExample = []
    nonMatchingAbstracts = 0
    wordNotFoundInFeatures = 0
    labelOfPhrase = ""

    for index in range(0, len(formatted_train)):
        newScore = []

        pharaseId = index
        phraseWordList = formatted_train[index]
        phraseClass = train_results_set['category'][index]
        scoreForThisPhrase = 0

        matchingWordList = set(newPhrase.split()) & set(phraseWordList.split())

        if not len(matchingWordList) == 0:
            for word in matchingWordList:
                wordIndex = vectorizer.vocabulary_.get(word)

                if wordIndex == None:
                    # print ("Word [%s] not found in tokenList for matching Abstract [%d]" % (word, index))
                    wordNotFoundInFeatures = wordNotFoundInFeatures + 1
                else:
                    wordScore = X_train_tfidf_array[index][wordIndex]
                    scoreForThisPhrase = scoreForThisPhrase + wordScore
        else:
            scoreForThisPhrase = 0
            nonMatchingAbstracts = nonMatchingAbstracts + 1

        newScore.append(pharaseId)
        newScore.append(scoreForThisPhrase)
        newScore.append(phraseClass)
        scoresPerExample.append(newScore)

    print("Non-Matching Abstracts = %d" % nonMatchingAbstracts)
    print("Number of time matching word not found in featureList = %d" % wordNotFoundInFeatures)

    topKMatches = sorted(scoresPerExample, key=lambda x: x[1], reverse=True)[:k]

    countOfMath = findLabelCount(label_math, topKMatches)
    countOfCS = findLabelCount(label_cs, topKMatches)
    countOfPhysics = findLabelCount(label_physics, topKMatches)
    countOfStat = findLabelCount(label_stat, topKMatches)

    countDictionary = {'math': countOfMath, 'cs': countOfCS, 'physics': countOfPhysics, 'stat': countOfStat}
    countValues = list(countDictionary.values())
    maximumCount = max(countValues)
    noOfMaximumCountLabels = countValues.count(maximumCount)

    if not noOfMaximumCountLabels == 1:
        print("++++++++++++ Equal no of neighbours for [%d]" % newPhraseId)
        print("++++++++++++ ", countDictionary)
        equalCountLabelsWithTotalScores = []

        for label, count in countDictionary.items():
            if count == maximumCount:
                total_labelScore = getTotalScoreOfLabel(label, topKMatches)
                newTotalScore = []
                newTotalScore.append(label)
                newTotalScore.append(total_labelScore)
                equalCountLabelsWithTotalScores.append(newTotalScore)

        max_score_label = sorted(equalCountLabelsWithTotalScores, key=lambda x: x[1], reverse=True)[:1]
        labelOfPhrase = max_score_label[0][0]
    else:
        for label, count in countDictionary.items():
            if count == maximumCount:
                labelOfPhrase = label

    return labelOfPhrase
# ======================================================================================


# =========================================================================
#           Main Method - Execution starts here...
# =========================================================================
if __name__ == "__main__":
    global noOfFeatures, rowsToConsider, remainingRows
    global label_math, label_cs, label_physics, label_stat

    label_math = "math"
    label_cs = "cs"
    label_stat = "stat"
    label_physics = "physics"

    # dividing the training set into two in order to create our own test set
    totalDocuments = 88639
    rowsToConsider = 70001
    remainingRows = totalDocuments - rowsToConsider

    # set the value for k and the number of features to be considered.
    k = 10
    noOfFeatures = 10000

    # load the training samples and the test abstracts
    train = loadTrainIn()
    test = loadTestIn()

    # --------------------------------------------------------------
    # Used for the case of splitting the training set into our own train-test set
    # --------------------------------------------------------------
    # trainingSet = train.head(rowsToConsider)
    # testSet = train.tail(remainingRows)
    # --------------------------------------------------------------

    trainDf = cleanSet(train, 0, len(train))
    testDf = cleanSet(test, 0, len(test))

    trainDf.to_csv("formatted_train_in.csv", index=False, header=False)
    testDf.to_csv("formatted_test_in.csv", index=False, header=False)

    # --------------------------------------------------------------
    # Used for the case of splitting the training set into our own train-test set
    # --------------------------------------------------------------
    # testDf = cleanSet(testSet, rowsToConsider, (rowsToConsider + remainingRows))
    # trainDf.to_csv("formatted_train_in_" + str(rowsToConsider) + ".csv", index=False, header=False)
    # testDf.to_csv("formatted_test_in_" + str(remainingRows) + ".csv", index=False, header=False)
    # --------------------------------------------------------------

    # Read the formatted/cleaned training abstracts from the saved file in the earlier step.
    formatted_train = pd.read_csv("formatted_train_in.csv", header=None, sep=",", usecols=[1])[1]

    # =========================================================================
    # Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
    print("Creating the bag of words...\n")

    if noOfFeatures == -1:
        vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None)
    else:
        vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                     max_features=noOfFeatures)

    ## using fit_transform() to fit the model and learn the vocabulary;
    ## then transforming our training data into feature vectors.
    train_data_features = vectorizer.fit_transform(formatted_train.values.astype('U'))

    ## Convert the result to a numpy array
    train_data_features = train_data_features.toarray()
    # =========================================================================

    print("Fetching Vocabulory and counting word frequencies...\n")
    vocab = vectorizer.get_feature_names()
    dist = np.sum(train_data_features, axis=0)

    vocabWordCount = []
    ## For each, print the vocabulary word and the number of times it appears in the training set
    for tag, count in zip(vocab, dist):
        newWordInVocab = []
        newWordInVocab.append(tag)
        newWordInVocab.append(count)
        vocabWordCount.append(newWordInVocab)

    vocabf = pd.DataFrame(vocabWordCount)
    print("Writing Vocabulory with word frequencies to file...\n")
    vocabf.to_csv("vocabulary_count.csv", index=False, header=False)
    # =========================================================================

    print("Creating feature array of td-idf...\n")
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(train_data_features)
    X_train_tfidf_array = X_train_tfidf.toarray()
    # =========================================================================

    train_results_set = pd.read_csv("../datasets/train_out.csv", header=0, sep=",")

    # --------------------------------------------------------------
    # Used for the case of splitting the training set into our own train-test set
    # --------------------------------------------------------------
    # train_set_results = results_set.head(rowsToConsider)
    # test_set_results = results_set.tail(remainingRows)
    # --------------------------------------------------------------

    # Loading the previously saved predictor-set
    predictor_set = pd.read_csv("formatted_test_in.csv", header=None, sep=",")
    predictedValues = []

    # for each of the abstract in the predictor set - predict its label
    for item in range(0, len(predictor_set)):
        newPhraseId = predictor_set[0][item]
        newPhrase = predictor_set[1][item]

        labelOfPhrase = findBestLabel(k, newPhraseId, newPhrase,
                                      formatted_train, train_results_set,
                                      vectorizer, X_train_tfidf_array)

        newPrediction = []
        newPrediction.append(newPhraseId)
        newPrediction.append(labelOfPhrase)
        predictedValues.append(newPrediction)
        print(newPhraseId, labelOfPhrase)
        print("==========================================================")

    # save predictions to a file
    pd.DataFrame(predictedValues).to_csv("final_predictions.csv", index=False, header=False)