__author__ = 'Ben'

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import gaussian_process as gp
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
import numpy as np
import math

# setting number of tree for the random forest
ntree = 300

final_test = pd.DataFrame.from_csv('C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/final_test.csv',
                                    header=0, index_col=None)

size = len(final_test)
daysOfWeek = final_test['DayOfWeek'].tolist()
trainPath = "C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/trainByDay/trainDay_%d.csv"
testPath = "C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/testByDay/testDay_%d.csv"
submission = pd.DataFrame(columns=['Id', 'Sales'])
progress = 0
identifiers = []
predictions = []
targetPredictions = []

# Create the random forest object which will include all the parameters for the fit
#forest = RandomForestRegressor(n_estimators=ntree)
tree = DecisionTreeRegressor()
gauss = gp.GaussianProcess()
forest = AdaBoostRegressor(n_estimators=ntree)

for i in set(daysOfWeek):
    train = pd.DataFrame.from_csv(trainPath % i, header=None, index_col=None)
    train = train.loc[(train[10] == 0) & (train[11] == 0) & (train[12] == 0)]
    train = train.drop(train.columns[[1, 10, 11, 12]], axis=1)
    test = train.ix[int(0.7*len(train))+1:, :]
    train = train.ix[0:int(0.7*len(train)), :]
    trainPromo = train.ix[train[5] == 1, :]
    trainNoPromo = train.ix[train[5] == 0, :]
    testPromo = test.ix[test[5] == 1, :]
    testNoPromo = test.ix[test[5] == 0, :]
    '''
    test = pd.DataFrame.from_csv(testPath % i, header=None, index_col=None)
    test = test.drop(test.columns[[9, 10, 11]], axis=1)
    testPromo = test.ix[test[4] == 1, :]
    testNoPromo = test.ix[test[4] == 0, :]
    #handling stores with promo in that day
    identifiers.extend(testPromo[0].values)
    testPromo = testPromo.ix[:, 1:testPromo.columns[-1]]
    testPromo.columns = range(testPromo.columns.size)
    '''
    # handling stores with promo in that day
    targetPredictions.extend(testPromo[2])
    testPromo = (testPromo.drop(testPromo.columns[[2, 3]], axis=1)).ix[:, 0:testPromo.columns[-1]]
    testPromo.columns = range(testPromo.columns.size)
    featuresPromo = (trainPromo.drop(trainPromo.columns[[2, 3]], axis=1)).ix[:, 0:trainPromo.columns[-1]]
    featuresPromo.columns = range(featuresPromo.columns.size)
    targetPromo = trainPromo[2]
    # fitting the forest with promo stores that day
    if len(featuresPromo) != 0:
        forestPromo = forest.fit(featuresPromo, targetPromo)
    promoImportance = forestPromo.feature_importances_
    #print(promoImportance)
    # predicting with the forest for promo stores that day
    if len(testPromo) != 0:
        predictions.extend(forestPromo.predict(testPromo))
    # handling stores with no promo in that day
    '''
    identifiers.extend(testNoPromo[0].values)
    testNoPromo = testNoPromo.ix[:, 1:testNoPromo.columns[-1]]
    testNoPromo.columns = range(testNoPromo.columns.size)
    '''
    targetPredictions.extend(testNoPromo[2])
    testNoPromo = (testNoPromo.drop(testNoPromo.columns[[2, 3]], axis=1)).ix[:, 0:testNoPromo.columns[-1]]
    testNoPromo.columns = range(testPromo.columns.size)
    featuresNoPromo = (trainNoPromo.drop(trainNoPromo.columns[[2, 3]], axis=1)).ix[:, 0:trainNoPromo.columns[-1]]
    featuresNoPromo.columns = range(featuresNoPromo.columns.size)
    targetNoPromo = trainNoPromo[2]
    #fitting the forest with no promo stores that day
    if len(featuresNoPromo) != 0:
        forestNoPromo = forest.fit(featuresNoPromo, targetNoPromo)
    noPromoImportance = forestNoPromo.feature_importances_
    #print(noPromoImportance)
    #predicting with the forest for no promo stores that day
    if len(testNoPromo) != 0:
        predictions.extend(forestNoPromo.predict(testNoPromo))
    #progress += len(test)
    print('finished predicting day ', i)
    #print('predicted store ', progress, ' over ', size)

# Kaggle error computation
sumRatio = 0
for i in range(len(predictions)):
    if predictions[i] != 0 and targetPredictions[i] != 0:
        sumRatio += (abs(predictions[i]-targetPredictions[i])/predictions[i])**2

# final error
error = math.sqrt(sumRatio/len(predictions))

print('PREDICTION COMPLETED, FINAL ERROR IS: ', error)

'''
### TEST ###
error = 0.062
'''

'''
submission['Id'] = identifiers
submission['Sales'] = predictions
submission = submission.sort(columns='Id', ascending=True, axis=0)
submission.to_csv("C:/Users/Ben/Google Drive/UIC/Machine Learning/ML project/Rossman_Data/forestPerDayAndPromo.csv", index=None)
'''