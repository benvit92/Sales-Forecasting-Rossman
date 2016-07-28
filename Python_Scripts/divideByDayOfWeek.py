__author__ = 'Ben'

import csv

trainPath = "C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/trainByDay/trainDay_%d.csv"
testPath = "C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/testByDay/testDay_%d.csv"

with open('C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/final_test.csv', 'r') as csvTest:
    testReader = csv.reader(csvTest, delimiter=',')
    next(testReader)
    for rowTest in csvTest:
        file = open(testPath % int(rowTest.split(',')[2]), 'a')
        file.write(rowTest)
        file.close()

print('test ended')

with open('C:/Users/Ben/Google Drive/UIC/Machine Learning/ML Project/Rossman_Data/final_train.csv', 'r') as csvTrain:
    trainReader = csv.reader(csvTrain, delimiter=',')
    next(trainReader)
    for rowTrain in csvTrain:
        file = open(trainPath % int(rowTrain.split(',')[1]), 'a')
        file.write(rowTrain)
        file.close()


