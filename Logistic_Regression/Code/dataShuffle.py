def dataShuffle(X,y):
  import numpy
  import random

  X_ = numpy.copy(X)
  y_ = numpy.copy(y)
  m,n = X_.shape

  s = (1, n)
  tempX = numpy.ones(s)

  my,ny = y.shape
  s = (1, ny)
  tempy = numpy.ones(s)

  assert m == my, "No. of data points in X and y are not equal."

  for i in range(m/2):
    p = random.randrange(0,m/2,1)
    q = random.randrange(m/2,m,1)

    tempX[0,:] = X_[p, :]
    X_[p, :] = X_[q,:]
    X_[q, :] = tempX[0,:]

    tempy[0, :] = y_[p, :]
    y_[p, :] = y_[q, :]
    y_[q, :] = tempy[0, :]

  for i in range(n):
    assert numpy.isclose(numpy.mean(X_[:,i]), numpy.mean(X[:,i]), 1e-05, 1e-08), "Mean of reshuffled X not equal original X."
    assert numpy.isclose(numpy.std(X_[:,i]), numpy.std(X[:,i]), 1e-05, 1e-08), "Standard dev of reshuffled X not equal original X."

  for i in range(ny):
    assert numpy.isclose(numpy.mean(y_[:,i]), numpy.mean(y[:,i]), 1e-05, 1e-08), "Mean of reshuffled y not equal original y."
    assert numpy.isclose(numpy.std(y_[:,i]), numpy.std(y[:,i]), 1e-05, 1e-08), "Standard dev of reshuffled y not equal original y."

  return X_,y_
