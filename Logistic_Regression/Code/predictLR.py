def predictLR(X, theta, threshold):
  import numpy
  import sigmoid as s
  import featureNormalize as fx

  X_ = fx.featureNormalize(X)
  return numpy.greater_equal(s.sigmoid(numpy.dot(X_, theta)), threshold)