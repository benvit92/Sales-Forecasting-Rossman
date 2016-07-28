__author__ = 'Ben'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import math
import random

# variables to set the number of folds, their size and the number of tree for the random forest
N = 15
foldSize = 100000
ntree = 300

final_test = pd.DataFrame.from_csv('C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/final_test.csv',
                                    header=0, index_col=None)

daysOfWeek = final_test['DayOfWeek'].tolist()
trainPath = "C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/trainByDay/trainDay_%d.csv"
testPath = "C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/testByDay/testDay_%d.csv"
submission = pd.DataFrame(columns=['Id', 'Sales'])
progress = 0
identifiers = []
finalResult = []
# list for final predictions
targetPredictions = []

# Create the random forest object which will include all the parameters for the fit
forest = RandomForestRegressor(n_estimators=ntree)

for i in set(daysOfWeek):
    # handling test and train
    train = pd.DataFrame.from_csv(trainPath % i, header=None, index_col=None)
    train = train.loc[(train[10] == 0) & (train[11] == 0) & (train[12] == 0)]
    train = train.drop(train.columns[[1, 10, 11, 12, 17]], axis=1)
    '''
    test = pd.DataFrame.from_csv(testPath % i, header=None, index_col=None)
    test = test.drop(test.columns[[2, 3, 9, 10, 11, 16]], axis=1)
    testPromo = test.ix[test[4] == 1, :].drop(4, axis=1)
    testNoPromo = test.ix[test[4] == 0, :].drop(4, axis=1)
    identifiers.extend(testPromo[0].values)
    identifiers.extend(testNoPromo[0].values)
    '''
    #test = train.ix[int(0.7*len(train))+1:, :]
    #train = train.ix[0:int(0.7*len(train)), :]
    rows = random.sample(train.index.tolist(), int(0.3*len(train)))
    test = train.ix[rows]
    train = train.drop(rows)
    testPromo = test.ix[test[4] == 1, :]
    testNoPromo = test.ix[test[4] == 0, :]
    '''
    testPromo = testPromo.ix[:, 1:testPromo.columns[-1]]
    testPromo.columns = range(testPromo.columns.size)
    testNoPromo = testNoPromo.ix[:, 1:testNoPromo.columns[-1]]
    testNoPromo.columns = range(testNoPromo.columns.size)
    '''
    targetPredictions.extend(testPromo[2])
    testPromo = (testPromo.drop(testPromo.columns[[2, 3, 5]], axis=1)).ix[:, 0:testPromo.columns[-1]]
    testPromo.columns = range(testPromo.columns.size)
    targetPredictions.extend(testNoPromo[2])
    testNoPromo = (testNoPromo.drop(testNoPromo.columns[[2, 3, 5]], axis=1)).ix[:, 0:testNoPromo.columns[-1]]
    testNoPromo.columns = range(testNoPromo.columns.size)
    #creating a dataframe where to store predictions
    predictions = pd.DataFrame()
    for j in range(N):
        # creating folds for the day in question
        foldPrediction = []
        name = 'prediction_'+str(j+1)
        fold = train.loc[np.random.choice(train.index, foldSize)]
        trainPromo = fold.ix[train[5] == 1, :].drop(5, axis=1)
        trainNoPromo = fold.ix[train[5] == 0, :].drop(5, axis=1)
        # handling stores with promo in that day
        featuresPromo = (trainPromo.drop(trainPromo.columns[[2, 3]], axis=1)).ix[:, 0:trainPromo.columns[-1]]
        featuresPromo.columns = range(featuresPromo.columns.size)
        targetPromo = trainPromo[2]
        # fitting the forest with promo stores that day
        if len(featuresPromo) != 0:
            forestPromo = forest.fit(featuresPromo, targetPromo)
        #predicting with the forest for promo stores that day
        if len(testPromo) != 0:
            foldPrediction.extend(forestPromo.predict(testPromo))
        #handling stores with no promo in that day
        featuresNoPromo = (trainNoPromo.drop(trainNoPromo.columns[[2, 3]], axis=1)).ix[:, 0:trainNoPromo.columns[-1]]
        featuresNoPromo.columns = range(featuresNoPromo.columns.size)
        targetNoPromo = trainNoPromo[2]
        #fitting the forest with no promo stores that day
        if len(featuresNoPromo) != 0:
            forestNoPromo = forest.fit(featuresNoPromo, targetNoPromo)
        #predicting with the forest for no promo stores that day
        if len(testNoPromo) != 0:
            foldPrediction.extend(forestNoPromo.predict(testNoPromo))
        predictions[name] = foldPrediction
        print('finished predicting fold ', j+1, ' over ', N, ' for day of week ', i)

    #getting the final prediction by keeping the one less distant from all the others
    final_prediction = []
    for k in range(len(predictions)):
        differences = []
        for j in range(len(predictions.columns)):
            difference = sum(predictions.loc[k].map(lambda x: abs(predictions.ix[k, j]-x)))
            differences.append(difference)
        final_prediction.append(predictions.ix[k, differences.index(min(differences))])
        #if (k % 10000) == 0:
            #print('final prediction progress for day ', i, ' is ', k+1, ' over ', len(predictions))

    # saving the final prediction
    finalResult.extend(final_prediction)

    # updating the counter
    # progress += len(test)
    # print('predicted store ', progress, ' over ', len(final_test))

# Kaggle error computation
sumRatio = 0
for i in range(len(finalResult)):
    if finalResult[i] != 0 and targetPredictions[i] != 0:
        sumRatio += ((finalResult[i]-targetPredictions[i])/finalResult[i])**2

# final error
error = math.sqrt(sumRatio/len(finalResult))
print('PREDICTION COMPLETED, FINAL ERROR IS: ', error)

'''
submission['Id'] = identifiers
submission['Sales'] = finalResult
submission = submission.sort(columns='Id', ascending=True, axis=0)
submission.to_csv("C:/Users/Ben/Google Drive/UIC/Machine Learning/ML project/Rossman_Data/forestBaggedPerDayAndPromo.csv", index=None)
'''

'''
### TESTS ###
folds = 1 size = 100000 => error = 0.0103
folds = 2 size = 100000 => error = 0.005
folds = 3 size = 100000 => error = 0.0018
folds = 4 size = 100000 => error = 0.0019
folds = 5 size = 100000 => error = 0.0084
folds = 6 size = 100000 => error = 0.00188
folds = 7 size = 100000 => error = 0.00112
folds = 8 size = 100000 => error = 0.00176
folds = 9 size = 100000 => error = 0.0161
folds = 10 size = 100000 => error = 0.0129
folds = 11 size = 100000 => error = 0.0023
folds = 12 size = 100000 => error = 0.00187
folds = 13 size = 100000 => error = 0.00226
folds = 14 size = 100000 => error = 0.00179
folds = 15 size = 100000 => error = 0.00154
'''