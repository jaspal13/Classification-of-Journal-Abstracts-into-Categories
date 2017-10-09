def calcPredictionErrorLR(X,y, theta, threshold):
  import predictLR as plr
  import numpy

  ypredict = plr.predictLR(X, theta, threshold)
  m,n = y.shape

  TP = sum(numpy.logical_and(ypredict, y))
  TN = sum(numpy.logical_not(numpy.logical_or(ypredict, y)))
  P = TP / float(sum(ypredict))
  R = TP / float(sum(y))
  Sp = TN / float(sum(numpy.logical_not(y)))

  return sum(numpy.logical_not(numpy.logical_xor(ypredict, y)))/float(m), P, R, Sp