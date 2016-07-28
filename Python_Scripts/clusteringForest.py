import pandas as pd
import statistics as st
import matplotlib.pyplot as plt
import sklearn.cluster as cl
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
import math

# set here the number of trees for the random forest
ntree = 500

# loading test set, train set and the store list
full_train = pd.DataFrame.from_csv('C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/final_train.csv',
                              header=0, index_col=None)
full_test = pd.DataFrame.from_csv('C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/final_test.csv',
                              header=0, index_col=None)

# removing zero values from train
full_train = full_train[full_train.Customers != 0]
full_train = full_train[full_train.Sales != 0]
full_train = full_train.loc[(full_train['IsPublicHoliday'] == 0) & (full_train['IsEasterHoliday'] == 0)\
              & (full_train['IsChristmasHoliday'] == 0)]

# creating a list of stores in the train set
trainStores = list(set(full_train['Store']))
statistics = pd.DataFrame(columns=['Store', 'AverageCustomers', 'CustomersDeviation', 'AverageSales', 'SalesDeviation'])

#################################################### STATISTICS ##########################################################

# computing mean of customers and sales for each store
for i in range(0, len(trainStores)):
    store = trainStores[i]
    meanCustomers = st.mean(full_train['Customers'][full_train['Store'] == store].values)
    customersDeviation = st.variance(full_train['Customers'][full_train['Store'] == store].values)
    meanSales = st.mean(full_train['Sales'][full_train['Store'] == store].values)
    salesDeviation = st.variance(full_train['Sales'][full_train['Store'] == store].values)
    statistics.loc[i] = [store, meanCustomers, customersDeviation, meanSales, salesDeviation]

print("statistics computation terminated... now starting with clustering")

######################################################## CLUSTERING #####################################################

# sorting and converting the result
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
# writing the cluster assignments on file
statistics.to_csv("C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/clustersAssignment.csv", index=None)

print("clustering terminated, beginning prediction now...")

###################################################### PREDICTION ########################################################

# predicting for each cluster
submission = pd.DataFrame(columns=['Id', 'Sales'])
progress = 0
identifiers = []
predictions = []
targetPredictions = []

# Create the random forest object which will include all the parameters for the fit
forest = RandomForestRegressor(n_estimators=ntree)
#forest = svm.SVR(kernel='linear')

#predicting dividing stores by promo
for i in range(1, n_clusters+1):
    clusterStores = statistics['Store'][statistics['Cluster'] == i]
    if (set(clusterStores).intersection(full_test['Store'])) != set():
        train = full_train[full_train.columns.tolist()][full_train['Store'].isin(clusterStores.values)]
        trainColumns = train.columns.tolist()
        # remove holiday columns from train
        trainColumns.remove('IsPublicHoliday')
        trainColumns.remove('IsEasterHoliday')
        trainColumns.remove('IsChristmasHoliday')
        # extracting sales and customers columns
        index = int(0.7*len(train))
        test = train.ix[index+1:, :]
        train = train.iloc[0:index, :]
        sales = trainColumns.pop(2)
        customers = trainColumns.pop(2)
        promoTrain = trainColumns.pop(3)
        trainPromo = train.ix[train[promoTrain] == 1, :]
        trainNoPromo = train.ix[train[promoTrain] == 0, :]
        testColumns = test.columns.tolist()
        #test = full_test[full_test.columns.tolist()][full_test['Store'].isin(clusterStores.values)]
        testColumns = test.columns.tolist()
        # remove holiday columns from test
        testColumns.remove('IsPublicHoliday')
        testColumns.remove('IsEasterHoliday')
        testColumns.remove('IsChristmasHoliday')
        testColumns.pop(2)
        testColumns.pop(2)
        promoTest = testColumns.pop(3)
        # extracting the id column from test
        # identifier = testColumns.pop(0)
        testPromo = test.ix[test[promoTest] == 1, :]
        testNoPromo = test.ix[test[promoTest] == 0, :]
        #handling stores with promo in that day
        # identifiers.extend(testPromo[identifier])
        targetPredictions.extend(testPromo['Sales'])
        testPromo = testPromo[testColumns]
        featuresPromo = trainPromo[trainColumns]
        targetPromo = trainPromo['Sales']
        #fitting the forest with promo stores that day
        forest.fit(featuresPromo, targetPromo)
        #predicting with the forest for promo stores that day
        predictions.extend(forest.predict(testPromo))
        #handling stores with no promo in that day
        # identifiers.extend(testNoPromo[identifier])
        targetPredictions.extend(testNoPromo['Sales'])
        testNoPromo = testNoPromo[testColumns]
        featuresNoPromo = trainNoPromo[trainColumns]
        targetNoPromo = trainNoPromo['Sales']
        #fitting the forest with no promo stores that day
        forest.fit(featuresNoPromo, targetNoPromo)
        #predicting with the forest for no promo stores that day
        predictions.extend(forest.predict(testNoPromo))
        # progress += (len(testPromo) + len(testNoPromo))
        # print('predicted store ', progress, ' over ', len(full_test))
        print('predicted cluster ', i, ' over ', n_clusters)

'''
# predictions without dividing stores by promo
for i in range(1, n_clusters+1):
    clusterStores = statistics['Store'][statistics['Cluster'] == i]
    #checking if stores in clusters are in the test set
    # if (set(clusterStores).intersection(full_test['Store'])) != set():
    train = full_train[full_train.columns.tolist()][full_train['Store'].isin(clusterStores)]
    trainColumns = train.columns.tolist()
    # remove holiday columns from train
    trainColumns.remove('IsPublicHoliday')
    trainColumns.remove('IsEasterHoliday')
    trainColumns.remove('IsChristmasHoliday')
    # pop sales column
    trainColumns.pop(2)
    # pop customers columns
    trainColumns.pop(2)
    # handling test
    # test = full_test[full_test.columns.tolist()][full_test['Store'].isin(clusterStores)]
    index = int(0.7*len(train))
    test = train.ix[index+1:, :]
    train = train.iloc[0:index, :]
    testColumns = test.columns.tolist()
    testColumns.remove('Sales')
    testColumns.remove('Customers')
    # remove holiday columns from test
    testColumns.remove('IsPublicHoliday')
    testColumns.remove('IsEasterHoliday')
    testColumns.remove('IsChristmasHoliday')
    # handling stores in that day
    # identifiers.extend(test[testColumns.pop(0)])
    targetPredictions.extend(test['Sales'])
    features = train[trainColumns]
    target = train['Sales']
    # fitting the forest with promo stores that day
    forest.fit(features, target)
    # predicting with the forest for promo stores that day
    predictions.extend(forest.predict(test[testColumns]))
    # handling stores with no promo in that day
    #progress += len(test)
    #print('predicted store ', progress, ' over ', len(full_test))
    print('predicted cluster ', i, ' over ', n_clusters)
'''

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
W/ PROMO DIVISION => error = 0.0904
W PROMO DIVISON => error = 0.0914
'''

'''
submission['Id'] = identifiers
submission['Sales'] = predictions
submission = submission.sort(columns='Id', ascending=True, axis=0)
submission.to_csv("C:/Users/Ben/Google Drive/UIC/Machine Learning/ML project/Rossman_Data/clusteringForest.csv", index=None)
'''