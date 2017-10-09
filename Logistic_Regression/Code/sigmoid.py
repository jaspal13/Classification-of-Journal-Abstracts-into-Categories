def sigmoid(X): 
  import numpy
  return numpy.true_divide(1, numpy.add(numpy.exp(numpy.multiply(X, -1.0)), 1))