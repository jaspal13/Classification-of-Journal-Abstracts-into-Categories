def featureNormalize(X):
  import numpy
  
  m,n = X.shape
  X_ = numpy.copy(X)
  
  assert numpy.isclose(numpy.mean(X_[:,0]), 1.0, 1e-05, 1e-08), "First coloumn of X is not 1."
  assert numpy.isclose(numpy.std(X_[:,0]), 0.0, 1e-05, 1e-08), "First coloumn of X is not 1."
  
  for i in range(1,n):
    mean = numpy.mean(X_[:,i])
    sigma = numpy.std(X_[:,i])
    
    X_[:,i] = numpy.subtract(X_[:,i], mean)
    X_[:,i] = numpy.true_divide(X_[:,i], sigma)
    
  return X_  