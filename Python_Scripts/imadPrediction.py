__author__ = 'Ben'

import pandas as pd
import statsmodels.api as sm
from datetime import datetime


final_test = pd.DataFrame.from_csv('C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/final_test.csv',
                                    header=0, index_col=None, )

size = len(final_test)
testStores = final_test['Store'].tolist()
trainPath = "C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/trainByStore/trainStore_%d.csv"
testPath = "C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/testByStore/testStore_%d.csv"
submission = pd.DataFrame(columns=['Id', 'Sales'])
progress = 0
identifiers = []
predictions = []

for i in set(testStores):
    #creating date objects for each store in the train
    train = pd.DataFrame.from_csv(trainPath % i, header=None, index_col=None)
    years = train[7]
    months = train[8]
    days = train[9]
    dates = []
    for j in range(len(train)):
        date = '-'.join([str(years[j]), str(months[j]), str(days[j])])
        dates.append(datetime.strptime(date, '%Y-%m-%d'))
    dates.sort()
    train.index = pd.Index(dates)
    #creating date for each store in the test
    test = pd.DataFrame.from_csv(testPath % i, header=None, index_col=None)
    years = test[6]
    months = test[7]
    days = test[8]
    dates = []
    for j in range(len(test)):
        date = '-'.join([str(years[j]), str(months[j]), str(days[j])])
        dates.append(date)
        #dates.append(datetime.strptime(date, '%Y-%m-%d'))
    test[1] = dates
    test.sort(1, inplace=True)
    identifiers.extend(test[0].values)
    test.index = pd.Index(test[1])
    #fitting for each store
    arma_model = sm.tsa.    ARMA(pd.TimeSeries(train[2], index=train.index), (3, 0)).fit()
    #predicting for each store
    predictions.extend(arma_model.predict(test.index[0], test.index[-1], dynamic=True))
    progress += len(test)
    print('predicted store ', progress, ' over ', size)

submission['Id'] = identifiers
submission['Sales'] = predictions
submission.sort(columns='Id', ascending=True, inplace=True)
submission.to_csv("C:/Users/Ben/Google Drive/UIC/Machine Learning/ML project/Rossman_Data/imadPrediction.csv", index=None)

