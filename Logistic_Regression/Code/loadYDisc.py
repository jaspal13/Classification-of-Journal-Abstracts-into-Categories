def loadYDisc(subject):
  import numpy

  name = subject + '.csv'
  return  numpy.matrix(numpy.loadtxt(name, delimiter=",")).transpose()
