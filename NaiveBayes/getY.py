import pandas as pd
import numpy as np


mathCategoryNumber = '1'
physicsCategoryNumber = '2'
statCategoryNumber = '3'
csCategoryNumber = '4'
result = pd.DataFrame(columns=('id', 'category'))
index = 0
print 'Generating Y Matrix...'
for row in pd.read_csv('../datasets/train_out.csv',sep=',', chunksize=1):
    idOld = row.ix[:,0:1].iloc[0]['id']
    categoryOld = row.ix[:,1:2].iloc[0]['category']
    #print idOld
    if(categoryOld == 'cs'):
        categoryOld = csCategoryNumber
    elif(categoryOld == 'math'):
        categoryOld = mathCategoryNumber
    elif(categoryOld == 'physics'):
        categoryOld = physicsCategoryNumber
    elif(categoryOld == 'stat'):
        categoryOld = statCategoryNumber
    result.loc[index]=[idOld,categoryOld]
    index += 1
result.to_csv('train_out_categorized1.csv', index=False)
print 'File Created:train_out_categorized1.csv'
