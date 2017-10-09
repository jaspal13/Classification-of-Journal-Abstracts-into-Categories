def getYLR(subject):
  import pandas
  import numpy

  df = pandas.read_csv('datasets/train_out.csv')

  Yin = df.as_matrix(columns=['category'])

  a,b = Yin.shape

  Yout = numpy.zeros((a, 1))

  for i in range(a):
    if subject == Yin[i]:
      Yout[i] = 1

  name = subject + '.csv'

  numpy.savetxt(name, Yout, delimiter=",")

  return Yout