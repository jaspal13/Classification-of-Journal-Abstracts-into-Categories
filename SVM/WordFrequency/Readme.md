0. The working directory should have the datasets folder
1. In the datasets folder, delete the rows with id-29023,60082,82989 on files train_in.csv,train_out.csv. These are garbage data with category as 'category' 
2.  run cleanData.py #takes 30-40 minutes or less
3. Set the noOfFeatures in first line of getX.py and run it.
Depending on the number of Features a file will be created which can take upto 2GB of space!
4. run getYLR.py#This will generate the Y Matrix
5. Set the noOfFeatures and kernel type in the first line of finalPredictions.py and run it
