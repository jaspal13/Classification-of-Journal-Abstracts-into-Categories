def costFunctionLR(X, y, theta, lmbda):
  import numpy
  import sigmoid as s
  m,n = X.shape

  h = s.sigmoid(numpy.dot(X,  theta))
  cost = numpy.subtract(h, y)

  y_neg = numpy.multiply(y, -1.0)
  cost_lr = numpy.multiply(y_neg, numpy.log(h))
  cost_lr = cost_lr - numpy.multiply(numpy.add(1, y_neg), numpy.log(numpy.subtract(1, h)))
  cost_lr = numpy.sum(cost_lr)
  cost_lr = numpy.true_divide(cost_lr, m)

  # set first term to zero for theta
  s=(n,1)
  mask = numpy.ones(s)
  mask.itemset(0, 0)
  mask = numpy.multiply(theta, mask)

  cost_reg = numpy.dot(mask.transpose(), mask)
  cost_reg = numpy.dot(cost_reg, lmbda)
  cost_reg = numpy.true_divide(cost_reg, numpy.dot(2, m))

  cost_tot = numpy.add(cost_lr, cost_reg)

  grad_err = numpy.dot(cost.transpose(), X)
  grad_err = grad_err.transpose()
  grad_err = numpy.true_divide(grad_err, m)

  grad_reg = numpy.dot(mask, lmbda)
  grad_reg = numpy.true_divide(grad_reg, m)

  grad = grad_reg + grad_err

  return cost_tot, grad





