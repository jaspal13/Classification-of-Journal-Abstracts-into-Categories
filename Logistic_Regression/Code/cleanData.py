def cleanData():
  import nltk
  import pandas
  import numpy
  import string
  from nltk.stem.snowball import SnowballStemmer
  from nltk.corpus import stopwords
  #from autocorrect import spell

  df = pandas.read_csv('datasets/train_in.csv')
  df1 = pandas.read_csv('datasets/test_in.csv')

  cleanData = pandas.DataFrame(columns=('id', 'abstract'))
  cleanData1 = pandas.DataFrame(columns=('id', 'abstract'))

  Xin = df.as_matrix(columns=['abstract'])
  Xin1 = df1.as_matrix(columns=['abstract'])

  a,b = Xin.shape
  a1,b1 = Xin1.shape

  stemmer = SnowballStemmer("english")

  stri = ' '
  for i in range(a):
    Xstr = numpy.array_str(Xin[i])
    Xstr = Xstr.lower();
    Xstr = Xstr.translate(None, string.punctuation)
    Xtok = nltk.word_tokenize(Xstr)

    Xtok_clean=[]

    for word in Xtok:
      if word not in stopwords.words('english'):
	Xtok_clean.append(stemmer.stem(word))

    stri = ' '.join(Xtok_clean)
    cleanData.loc[i] = [i, stri]

    if i%1000 == 0:
      print "%s" %(i)

  stri = ' '
  for i in range(a1):
    Xstr = numpy.array_str(Xin1[i])
    Xstr = Xstr.lower();
    Xstr = Xstr.translate(None, string.punctuation)
    Xtok = nltk.word_tokenize(Xstr)

    Xtok_clean=[]

    for word in Xtok:
      if word not in stopwords.words('english'):
	Xtok_clean.append(stemmer.stem(word))

    stri = ' '.join(Xtok_clean)
    cleanData1.loc[i] = [i, stri]

    if i%1000 == 0:
      print "%s" %(i)

  cleanData.to_csv('train_in_clean.csv')
  cleanData1.to_csv('test_in_clean.csv')

  return cleanData, cleanData1

