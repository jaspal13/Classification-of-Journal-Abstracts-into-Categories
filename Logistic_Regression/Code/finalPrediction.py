def finalPrediction(nFeatures, lmbda, iters):
  import pandas
  import numpy
  import loadYDisc as ly
  import loadXDisc as lx
  import trainLR as tlr
  import featureNormalize as fx
  import sigmoid as s
  import calcPredictionErrorLR as cpe

  df = pandas.DataFrame(columns=('id', 'category'))

  Xtrain, Xtest = lx.loadXDisc(nFeatures)

  Ycs = ly.loadYDisc('cs')
  Ymath = ly.loadYDisc('math')
  Ystat = ly.loadYDisc('stat')
  Yphysics = ly.loadYDisc('physics')

  print "Data Load complete!!"

  statistics = numpy.matrix(numpy.zeros((4,4)))

  thetaCs = tlr.trainLR(Xtrain, Ycs, lmbda, 0.1, 1e-015, iters, False)
  print "Training CS complete"
  A, P, R, S = cpe.calcPredictionErrorLR(Xtrain, Ycs, thetaCs, 0.5)
  statistics[0,0] = A
  statistics[0,1] = P
  statistics[0,2] = R
  statistics[0,3] = S


  thetaMath = tlr.trainLR(Xtrain, Ymath, lmbda, 0.1, 1e-015, iters, False)
  print "Training math complete"
  A, P, R, S = cpe.calcPredictionErrorLR(Xtrain, Ymath, thetaMath, 0.5)
  statistics[1,0] = A
  statistics[1,1] = P
  statistics[1,2] = R
  statistics[1,3] = S

  thetaStat = tlr.trainLR(Xtrain, Ystat, lmbda, 0.1, 1e-015, iters, False)
  print "Training stat complete"
  A, P, R, S = cpe.calcPredictionErrorLR(Xtrain, Ystat, thetaStat, 0.5)
  statistics[2,0] = A
  statistics[2,1] = P
  statistics[2,2] = R
  statistics[2,3] = S

  thetaPhysics = tlr.trainLR(Xtrain, Yphysics, lmbda, 0.1, 1e-015, iters, False)
  print "Training physics complete"
  A, P, R, S = cpe.calcPredictionErrorLR(Xtrain, Yphysics, thetaPhysics, 0.5)
  statistics[3,0] = A
  statistics[3,1] = P
  statistics[3,2] = R
  statistics[3,3] = S

  numpy.savetxt('statistics.csv', statistics, delimiter=",")

  XtestNorm = fx.featureNormalize(Xtest)

  YoutCS = s.sigmoid(numpy.dot(XtestNorm, thetaCs))
  YoutMath = s.sigmoid(numpy.dot(XtestNorm, thetaMath))
  YoutStat = s.sigmoid(numpy.dot(XtestNorm, thetaStat))
  YoutPhysics = s.sigmoid(numpy.dot(XtestNorm, thetaPhysics))

  Yout = numpy.hstack((YoutCS, YoutMath))
  Yout = numpy.hstack((Yout, YoutStat))
  Yout = numpy.hstack((Yout, YoutPhysics))

  print "Final predictions complete."
  a,b = Xtest.shape

  Yarg = Yout.argmax(axis=1)

  for i in range(a):
    if (Yarg[i] == 0):
      stri = 'cs'
    elif (Yarg[i] == 1):
      stri = 'math'
    elif (Yarg[i] == 2):
      stri = 'stat'
    else:
      stri = 'physics'


    df.loc[i] = [i, stri]

  df.to_csv('finalPrediction.csv', columns=['id', 'category'], index=False)
