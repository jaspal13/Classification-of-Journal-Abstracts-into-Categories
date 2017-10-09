def getXLR(nFeatures, featureOffset = 3):
  import pandas
  import nltk
  import numpy
  from collections import Counter

  df = pandas.read_csv('train_in_clean.csv')
  df1 = pandas.read_csv('test_in_clean.csv')

  Xin = df.as_matrix(columns=['abstract'])
  Xin1 = df1.as_matrix(columns=['abstract'])

  a,b = Xin.shape
  a1,b1 = Xin1.shape

  stri = ' '

  for i in range(a):
    stri = stri + ' '.join(Xin[i])

  Xtok = nltk.word_tokenize(stri)

  count = Counter(Xtok)

  stri = ' '

  for i in range(a1):
    stri = stri + ' '.join(Xin1[i])

  Xtok1 = nltk.word_tokenize(stri)

  count1 = Counter(Xtok1)

  fList=[]

  fList = count.most_common((nFeatures + featureOffset))

  Xf = numpy.zeros((a, nFeatures))
  Xf1 = numpy.zeros((a1, nFeatures))

  for i in range(a):
    if i%1000 == 0:
      print "%s" %(i)
    tok = nltk.word_tokenize(numpy.array_str(Xin[i]))
    for j in range(nFeatures):
      p,q = fList[j + featureOffset]
      if p in tok:
	Xf[i,j] = int(Xf[i,j]) + int(1)

  for i in range(a1):
    if i%1000 == 0:
      print "%s" %(i)
    tok = nltk.word_tokenize(numpy.array_str(Xin1[i]))
    for j in range(nFeatures):
      p,q = fList[j + featureOffset]
      if p in tok:
	Xf1[i,j] = int(Xf1[i,j]) + int(1)

  Xf = numpy.hstack((numpy.ones((a,1)), Xf))
  Xf1 = numpy.hstack((numpy.ones((a1,1)), Xf1))

  name = 'X_train_in' + str(nFeatures) + '.csv'
  numpy.savetxt(name, Xf, delimiter=",")

  name = 'X_test_in' + str(nFeatures) + '.csv'
  numpy.savetxt(name, Xf1, delimiter=",")


  return count, count1, Xf, Xf1
