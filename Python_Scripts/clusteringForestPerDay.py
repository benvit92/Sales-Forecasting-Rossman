__author__ = 'Ben'

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import gaussian_process as gp
import statistics as st
from sklearn import cluster as cl
import sys

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

# Create the random forest object which will include all the parameters for the fit
forest = RandomForestRegressor(n_estimators=ntree)

for i in set(daysOfWeek):
    trainDay = pd.DataFrame.from_csv(trainPath % i, header=None, index_col=None)
    testDay = pd.DataFrame.from_csv(testPath % i, header=None, index_col=None)
    # removing zero values from train
    trainDay = trainDay[trainDay[2] != 0]
    trainDay = trainDay[trainDay[3] != 0]
    # creating a list of stores in the train set
    trainStores = list(set(trainDay[0]))
    statistics = pd.DataFrame(columns=['Store', 'AverageCustomers', 'CustomersDeviation', 'AverageSales', 'SalesDeviation'])
    # computing mean of customers and sales for each store
    for j in range(0, len(trainStores)):
        store = trainStores[j]
        meanCustomers = st.mean(trainDay.ix[trainDay[0] == store, 3].values)
        customersDeviation = st.variance(trainDay.ix[trainDay[0] == store, 3].values)
        meanSales = st.mean(trainDay.ix[trainDay[0] == store, 2].values)
        salesDeviation = st.variance(trainDay.ix[trainDay[0] == store, 2].values)
        statistics.loc[j] = [store, meanCustomers, customersDeviation, meanSales, salesDeviation]

    # converting the result
    statistics = statistics.sort(columns='Store', ascending=True, axis=0)
    statistics_np = statistics.as_matrix(['AverageCustomers', 'CustomersDeviation', 'AverageSales', 'SalesDeviation'])

    # running clustering algorithm
    clusters = cl.AffinityPropagation().fit(statistics_np)
    cluster_centers_indices = clusters.cluster_centers_indices_
    labels = clusters.labels_
    n_clusters = len(cluster_centers_indices)

    print('number of clusters is ', n_clusters)

    # adding cluster to the statistics dataframe
    statistics['Cluster'] = clusters.labels_ + 1

    # performing prediction per cluster according to that specific day
    for j in range(1, n_clusters+1):
        clusterStores = statistics['Store'][statistics['Cluster'] == j]
        if (set(clusterStores).intersection(testDay[1])) != set():
            train = trainDay[trainDay.columns.tolist()][trainDay[0].isin(clusterStores.values)]
            test = testDay[testDay.columns.tolist()][testDay[1].isin(clusterStores.values)]
            identifiers.extend(test[0].values)
            test = test.ix[:, 1:test.columns[-1]]
            test.columns = range(test.columns.size)
            features = (train.drop(train.columns[[2, 3]], axis=1)).ix[:, 0:train.columns[-1]]
            features.columns = range(features.columns.size)
            target = train[2]
            #fitting for each store
            forest = forest.fit(features, target)
            #predicting for each store
            predictions.extend(forest.predict(test))
            progress += len(test)
            print('predicted ', progress, ' lines over ', size)

'''
    for j in range(1, n_clusters+1):
        clusterStores = statistics['Store'][statistics['Cluster'] == j]
        if (set(clusterStores).intersection(testDay[1])) != set():
            trainPromo = train.ix[train[4] == 1, :]
            trainNoPromo = train.ix[train[4] == 0, :]
            testPromo = test.ix[test[3] == 1, :]
            testNoPromo = test.ix[test[3] == 0, :]
            #handling stores with promo in that day
            identifiers.extend(testPromo[0].values)
            testPromo = testPromo.ix[:, 1:testPromo.columns[-1]]
            testPromo.columns = range(testPromo.columns.size)
            featuresPromo = (trainPromo.drop(trainPromo.columns[[2, 3]], axis=1)).ix[:, 0:trainPromo.columns[-1]]
            featuresPromo.columns = range(featuresPromo.columns.size)
            targetPromo = trainPromo[2]
            #fitting the forest with promo stores that day
            forestPromo = forest.fit(featuresPromo, targetPromo)
            #predicting with the forest for promo stores that day
            predictions.extend(forestPromo.predict(testPromo))
            #handling stores with no promo in that day
            identifiers.extend(testNoPromo[0].values)
            testNoPromo = testNoPromo.ix[:, 1:testNoPromo.columns[-1]]
            testNoPromo.columns = range(testNoPromo.columns.size)
            featuresNoPromo = (trainNoPromo.drop(trainNoPromo.columns[[2, 3]], axis=1)).ix[:, 0:trainNoPromo.columns[-1]]
            featuresNoPromo.columns = range(featuresNoPromo.columns.size)
            targetNoPromo = trainNoPromo[2]
            #fitting the forest with no promo stores that day
            forestNoPromo = forest.fit(featuresNoPromo, targetNoPromo)
            #predicting with the forest for no promo stores that day
            predictions.extend(forestNoPromo.predict(testNoPromo))
            progress += len(test)
            print('predicted store ', progress, ' over ', size)
'''

submission['Id'] = identifiers
submission['Sales'] = predictions
submission = submission.sort(columns='Id', ascending=True, axis=0)
submission.to_csv("C:/Users/Ben/Google Drive/UIC/Machine Learning/ML project/Rossman_Data/clusteringForestPerDay.csv", index=None)
