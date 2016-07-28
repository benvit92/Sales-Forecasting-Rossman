__author__ = 'Ben'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.neighbors import NearestNeighbors
import statistics
import math

# loading test set, train set and the store list
train = pd.DataFrame.from_csv('C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/final_train.csv',
                              header=0, index_col=None, )
# test = pd.DataFrame.from_csv('C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/final_test.csv',
#                              header=0, index_col=None)
test = train.ix[int(0.7*len(train))+1:, :]
train = train.ix[0:int(0.7*len(train)), :]
targetPredictions = test['Sales'].tolist()
test = test.drop(['Sales', 'Customers'], 1)

# index = test['Id']

# variables to set the number of folds, their size and the number of tree for the random forest
N = 5
size = 700000
ntree = 150

#creating a dataframe where to store predictions
predictions = pd.DataFrame()
# Create the random forest object which will include all the parameters for the fit
forest = RandomForestRegressor(n_estimators=ntree)

# sampling the dataset N times and storing each prediction into the predictions list
for i in range(N):
    fold = (train.loc[np.random.choice(train.index, size, replace=False)])
    print('created fold ', i+1)
    target = fold['Sales']
    features = (fold.drop(['Sales', 'Customers'], 1)).ix[:, 0:fold.columns.size]
    forest = forest.fit(features, target)
    #predicting with each classifier
    prediction = forest.predict(test.ix[:, 0:test.columns.size])
    print('classifier ', i+1, ' has predicted')
    name = 'prediction_'+str(i+1)
    predictions[name] = prediction

#getting the final prediction by keeping the one less distant from all the others
final_prediction = []
for i in range(len(predictions)):
    differences = []
    #if test['Open'][i] != 0:
    for j in range(len(predictions.columns)):
         difference = sum(predictions.loc[i].map(lambda x: abs(predictions.ix[i, j]-x)))
            #for k in range(len(predictions.columns)):
                #difference += abs(predictions.ix[i, j]-predictions.ix[i, k])
         differences.append(difference)
    final_prediction.append(predictions.ix[i, differences.index(min(differences))])
    #else:
        #final_prediction.append(0)
    if (i % 10000) == 0:
        print('final prediction progress is ', i+1, ' over ', len(predictions))

# Kaggle error computation
sumRatio = 0
for i in range(len(final_prediction)):
    if targetPredictions[i] != 0:
        sumRatio += ((final_prediction[i]-targetPredictions[i])/final_prediction[i])**2

# final error
error = math.sqrt(sumRatio/len(final_prediction))
print('PREDICTION COMPLETED, FINAL ERROR IS: ', error)

'''
### TEST ###
error = 0.134
'''

'''
#creating final dataframe for submission
submission = pd.DataFrame(columns=['Id', 'Sales'])
submission['Id'] = test['Id']
submission['Sales'] = prediction
#writing prediction to csv
submission.to_csv("C:/Users/Ben/Google Drive/UIC/Machine Learning/ML project/Rossman_Data/random.csv", index=None)
'''