__author__ = 'Ben'

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import math
import pylab as pl
import numpy as np
import random

ntree = 150

final_test = pd.DataFrame.from_csv('C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/final_test.csv',
                                    header=0, index_col=None, )
final_train = pd.DataFrame.from_csv('C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/final_train.csv',
                                    header=0, index_col=None, )

size = len(final_test)
daysOfWeek = final_test['DayOfWeek'].tolist()
trainPath = "C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/trainByDay/trainDay_%d.csv"
testPath = "C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/testByDay/testDay_%d.csv"
submission = pd.DataFrame(columns=['Id', 'Sales'])
progress = 0
identifiers = []
predictions = []
targetPredictions = []

#features list
featuresList = final_train.columns.tolist()
# removing sales and customers since not used
featuresList.remove('Sales')
featuresList.remove('Customers')
featuresList.remove('DayOfWeek')
finalImportance = [0]*len(featuresList)

# Create the random forest object which will include all the parameters for the fit
forest = RandomForestRegressor(n_estimators=ntree)

for i in set(daysOfWeek):
    train = pd.DataFrame.from_csv(trainPath % i, header=None, index_col=None)
    # train = train.loc[(train[10] == 0) & (train[11] == 0) & (train[12] == 0)]
    # train = train.drop(train.columns[[10, 11, 12]], axis=1)
    rows = random.sample(train.index.tolist(), int(0.3*len(train)))
    test = train.ix[rows]
    train = train.drop(rows)
    '''
    test = train.ix[int(0.7*len(train))+1:, :]
    train = train.ix[0:int(0.7*len(train)), :]
    '''
    targetPredictions.extend(test[2])
    '''
    test = pd.DataFrame.from_csv(testPath % i, header=None, index_col=None)
    identifiers.extend(test[0].values)
    test = test.ix[:, 1:test.columns[-1]]
    test.columns = range(test.columns.size)
    '''
    test = (test.drop(test.columns[[1, 2, 3]], axis=1)).ix[:, 0:test.columns[-1]]
    test.columns = range(test.columns.size)
    features = (train.drop(train.columns[[1, 2, 3]], axis=1)).ix[:, 0:train.columns[-1]]
    features.columns = range(features.columns.size)
    target = train[2]
    #fitting for each store
    forest = forest.fit(features, target)
    importance = forest.feature_importances_
    for j in range(len(importance)):
        finalImportance[j] += importance[j]
    #predicting for each store
    predictions.extend(forest.predict(test))
    # progress += len(test)
    # print('predicted store ', progress, ' over ', size)
    print('predicted day ', i)

# Kaggle error computation
sumRatio = 0
for i in range(len(predictions)):
    if targetPredictions[i] != 0:
        sumRatio += (abs(predictions[i]-targetPredictions[i])/predictions[i])**2

# final error
error = math.sqrt(sumRatio/len(predictions))

print('PREDICTION COMPLETED, FINAL ERROR IS: ', error)

### BARPLOT ###

fig = pl.figure()
ax = pl.subplot(111)
width = 1.2
ax.bar(range(len(featuresList)), finalImportance, width=width)
ax.set_xticks(np.arange(len(featuresList)) + width/2)
ax.set_xticklabels(featuresList, rotation=90)
pl.show()

'''
### TEST ###
error = 0.141
'''

'''
submission['Id'] = identifiers
submission['Sales'] = predictions
submission = submission.sort(columns='Id', ascending=True, axis=0)
submission.to_csv("C:/Users/Ben/Google Drive/UIC/Machine Learning/ML project/Rossman_Data/forestPerDay.csv", index=None)
'''