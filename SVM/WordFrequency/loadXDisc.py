def loadXDisc(nFeatures):
  import numpy

  name = 'X_train_in' + str(nFeatures) + '.csv'
  Xf = numpy.matrix(numpy.loadtxt(name, delimiter=","))

  name = 'X_test_in' + str(nFeatures) + '.csv'
  Xf1 = numpy.matrix(numpy.loadtxt(name, delimiter=","))

  return Xf, Xf1
