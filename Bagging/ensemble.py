def ensemble():
  import pandas

  df = pandas.read_csv('ensemble.csv')
  a,b = df.shape
  output = pandas.DataFrame(columns=('id', 'category'))

  for i in range(a):
    countMath = 0
    countStat = 0
    countPhy = 0
    countCs = 0
    for j in range(b):
      a = df.loc[i][j]
      if a == 'math':
	countMath = countMath + 1
      if a == 'physics':
	countPhy = countPhy + 1
      if a == 'stat':
	countStat = countStat + 1
      if a == 'cs':
	countCs = countCs + 1
    if (countMath > countPhy and countMath > countStat and countMath > countCs):
      subject = 'math'
    elif (countPhy > countMath and countPhy > countStat and countPhy > countCs):
      subject = 'physics'
    elif (countStat > countMath and countStat > countPhy and countStat > countCs):
      subject = 'stat'
    else:
      subject = 'cs'
    output.loc[i] = [i, subject]

  output.to_csv('ensemble_out.csv', columns=['id', 'category'], index=False)