0. The working directory should have the datasets folder
1. Open ipython shell and type the following
2. run cleanData.py
3. cleanData() #takes 30-40 minutes or less
4. run getXLR.py
5. getXLR(500) #extract 500 features for train and prediction, save them to disk
6. run getYLR.py
7. getYLR('cs') # cs vs all as Y
8. getYLR('math') # math vs all
9. getYLR('stat') # stat vs all
10. getYLR('physics') # physics vs all
11. run finalPrediction.py
12. finalPrediction(500, 0, 10000) #500 features, 0 regularization, 100000 iterations, could take a while
13. Find the final predictions in finalPrediction.csv
