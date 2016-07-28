__author__ = 'Ben'

import pandas as pd
import numpy as np
from datetime import datetime
import csv

# loading test set, train set and the store list
train = pd.DataFrame.from_csv('C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/train.csv',
                              header=0, index_col=None,)
test = pd.DataFrame.from_csv('C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/test.csv',
                              header=0, index_col=None)
store = pd.DataFrame.from_csv('C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/store.csv',
                              header=0, index_col=None)

#converting date from string into datetime for train and test
train['Date'] = train['Date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))
test['Date'] = test['Date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d'))

#Adding Year, Day and Month as three separate features to train and set
train['Year'] = train['Date'].map(lambda x: x.year)
train['Month'] = train['Date'].map(lambda x: x.month)
train['Day'] = train['Date'].map(lambda x: x.day)
test['Year'] = test['Date'].map(lambda x: x.year)
test['Month'] = test['Date'].map(lambda x: x.month)
test['Day'] = test['Date'].map(lambda x: x.day)

#handling categorical attributes(by creating a binary feature for each category)
train['IsPublicHoliday'] = train['StateHoliday'].map(lambda x: 1 if x=='a' else 0)
train['IsEasterHoliday'] = train['StateHoliday'].map(lambda x: 1 if x=='b' else 0)
train['IsChristmasHoliday'] = train['StateHoliday'].map(lambda x: 1 if x=='c' else 0)

test['IsPublicHoliday'] = test['StateHoliday'].map(lambda x: 1 if x=='a' else 0)
test['IsEasterHoliday'] = test['StateHoliday'].map(lambda x: 1 if x=='b' else 0)
test['IsChristmasHoliday'] = test['StateHoliday'].map(lambda x: 1 if x=='c' else 0)

store['isTypeA'] = store['StoreType'].map(lambda x: 1 if x=='a' else 0)
store['isTypeB'] = store['StoreType'].map(lambda x: 1 if x=='b' else 0)
store['isTypeC'] = store['StoreType'].map(lambda x: 1 if x=='c' else 0)
store['isTypeD'] = store['StoreType'].map(lambda x: 1 if x=='d' else 0)

store['Assortment'] = store['Assortment'].map(lambda x: 1 if x == 'a' else (2 if x == 'b' else 3))

#store['isAssortmentBasic'] = store['Assortment'].map(lambda x: 1 if x=='a' else 0)
#store['isAssortmentExtra'] = store['Assortment'].map(lambda x: 1 if x=='b' else 0)
#store['isAssortmentExtended'] = store['Assortment'].map(lambda x: 1 if x=='c' else 0)

#filling NaN for stores that do not have promotions active
store.fillna(0, inplace=True)
test.fillna(0, inplace=True)
train.fillna(0, inplace=True)

#creating dictionary to convert months from string to relative number
monthsDict ={'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
             'Jul': 7, 'Aug': 8, 'Sept': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

#converting from string to list of months with relative number and then creating one feature for each renewal month
store['Promo2Month1'] = store['PromoInterval'].map(lambda x: 0)
store['Promo2Month2'] = store['PromoInterval'].map(lambda x: 0)
store['Promo2Month3'] = store['PromoInterval'].map(lambda x: 0)
store['Promo2Month4'] = store['PromoInterval'].map(lambda x: 0)
for i in range(len(store)):
    if store['PromoInterval'][i] == 0:
        continue
    else:
        monthsList = store['PromoInterval'][i].split(",")
        for j in range(len(monthsList)):
            monthsList[j] = monthsDict[monthsList[j]]
        store['Promo2Month1'][i] = monthsList[0]
        store['Promo2Month2'][i] = monthsList[1]
        store['Promo2Month3'][i] = monthsList[2]
        store['Promo2Month4'][i] = monthsList[3]

#dropping the column not necessary anymore from train and test
train = train.drop(['Date', 'StateHoliday'], 1)
test = test.drop(['Date', 'StateHoliday'], 1)
store = store.drop(['StoreType', 'PromoInterval'], 1)

#writing preprocessed data to csv
train.to_csv("C:/Users/Ben/Google Drive/UIC/Machine Learning/ML project/Rossman_Data/train_preprocessed.csv", index=None)
test.to_csv("C:/Users/Ben/Google Drive/UIC/Machine Learning/ML project/Rossman_Data/test_preprocessed.csv", index=None)
store.to_csv("C:/Users/Ben/Google Drive/UIC/Machine Learning/ML project/Rossman_Data/store_preprocessed.csv", index=None)

#creating two dataframe with same number of rows of train and test with the relative store info
train_stores = pd.DataFrame(columns=store.columns.values.tolist())
test_stores = pd.DataFrame(columns=store.columns.values.tolist())
train_stores.to_csv('C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/train_stores.csv', index=None)
test_stores.to_csv('C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/test_stores.csv', index=None)


csvTrain = csv.reader(open('C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/train_preprocessed.csv'),
                      delimiter=',')
csvTest = csv.reader(open('C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/test_preprocessed.csv'),
                      delimiter=',', )

#progress counter
i = 0

with open('C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/test_stores.csv', 'w') as csvTestStores:
    testWriter = csv.writer(csvTestStores, delimiter=',', lineterminator='\n')
    for rowTest in csvTest:
        with open('C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/store_preprocessed.csv', 'r')\
                as csvStore:
                storeReader = csv.reader(csvStore, delimiter=',')
                for rowStore in storeReader:
                    if rowTest[1] == rowStore[0]:
                        testWriter.writerow(rowStore)

with open('C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/train_stores.csv', 'w') as csvTrainStores:
    trainWriter = csv.writer(csvTrainStores, delimiter=',', lineterminator='\n')
    for rowTrain in csvTrain:
        with open('C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/store_preprocessed.csv', 'r')\
                as csvStore:
                storeReader = csv.reader(csvStore, delimiter=',')
                for rowStore in storeReader:
                    if rowTrain[0] == rowStore[0]:
                        trainWriter.writerow(rowStore)
        if i%50000 == 0:
            print('processed up to', i+1, 'over ', len(train))
        i += 1

train_stores = pd.DataFrame.from_csv('C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/train_stores.csv',
                              header=0, index_col=None)
test_stores = pd.DataFrame.from_csv('C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/test_stores.csv',
                              header=0, index_col=None)

#dropping the store column
train_stores = train_stores.drop('Store', 1)
test_stores = test_stores.drop('Store', 1)

#creating final test and final train and test by adding the stores info
final_train = pd.concat([train, train_stores], axis=1)
final_test = pd.concat([test, test_stores], axis=1)

#writing the preprocessed train, test and store dataframes
final_train.to_csv("C:/Users/Ben/Google Drive/UIC/Machine Learning/ML project/Rossman_Data/final_train.csv", index=None)
final_test.to_csv("C:/Users/Ben/Google Drive/UIC/Machine Learning/ML project/Rossman_Data/final_test.csv", index=None)