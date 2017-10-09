def trainLR(X, y, lmbda, alpha, eps, max_iter, debug=False):
  import numpy
  import featureNormalize as fx
  import dataShuffle as ds
  import costFunctionLR as cf
  import gDesc1D as gd

  X_,y_ = ds.dataShuffle(X,y)
  X_ = fx.featureNormalize(X_)

  return gd.gDesc1D(X_, y_, lmbda, cf.costFunctionLR, alpha, eps, max_iter, debug)




