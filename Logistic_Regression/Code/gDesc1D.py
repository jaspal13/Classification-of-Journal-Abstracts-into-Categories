def gDesc1D(X, y, lmbda, costFunction, alpha, eps, max_iters, debug=False):
  import numpy

  # get dimension of X
  m,n = X.shape

  # initialize theta with random values
  theta = numpy.random.rand(n, 1)

  # run the loop for max_iters or util dJ tends to zero.
  for k in range(max_iters):
    J, dJ = costFunction(X, y, theta, lmbda)
    if debug:
      print "norm_dj=%s J=%s k=%s %s\n" %(numpy.linalg.norm(dJ),J,k, theta)
    # Check if dJ tends to zero.
    if numpy.linalg.norm(dJ) < eps:
      break
    delta = numpy.dot(dJ, alpha)
    theta = numpy.subtract(theta,delta)

    if k == max_iters/20:
      print "5 percent complete. norm_dj=%s J=%s k=%s %s\n" %(numpy.linalg.norm(dJ),J,k, theta)

    elif k == max_iters/10:
      print "10 percent complete. norm_dj=%s J=%s k=%s %s\n" %(numpy.linalg.norm(dJ),J,k, theta)

    elif k == max_iters/5:
      print "20 percent complete. norm_dj=%s J=%s k=%s %s\n" %(numpy.linalg.norm(dJ),J,k, theta)

    elif k == max_iters/2:
      print "50 percent complete. norm_dj=%s J=%s k=%s %s\n" %(numpy.linalg.norm(dJ),J,k, theta)

    elif k == (3*max_iters)/4:
      print "75 percent complete. norm_dj=%s J=%s k=%s %s\n" %(numpy.linalg.norm(dJ),J,k, theta)


  return theta

