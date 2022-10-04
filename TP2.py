
from distutils.util import copydir_run_2to3
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from pyexcel.cookbook import merge_all_to_a_book
import csv
import numpy
import sklearn
from datetime import *
import os
import glob
import csv

# Chargement des données
data = csv.reader(open("SeoulBikedata.csv", "r"),
                  delimiter=",")

data = list(data)
# delete la premiere ligne
data = numpy.delete(data, 0, 0)

# On extraie la colonne des dates
dates = [i[0] for i in data]


def str_date_to_datatime(str_date):
    return datetime.strptime(str_date, '%d/%m/%Y')


# on transforme notre tableau de dates de type string en dates de type date
for i in range(0, int(len(dates))):
    dates[i] = str_date_to_datatime(dates[i])

# transforms all date in days of the week


def date_to_days(tab_date):
    day = numpy.zeros(int(len(tab_date)))
    for i in range(0, int(len(tab_date))):
        day[i] = tab_date[i].weekday()
    return day


# replace first column by days of the week
dates = date_to_days(dates)
for i in range(0, int(len(dates))):
    data[i][0] = dates[i]

# replace yes from column 10 by 1 and no by 0
for i in range(0, int(len(data))):

    if data[i][13] == 'Yes':
        data[i][13] = 1
    else:
        data[i][13] = 0

    if data[i][12] == 'No Holiday':
        data[i][12] = 1
    else:
        data[i][12] = 0

    if data[i][11] == 'Winter':
        data[i][11] = 4
    elif data[i][11] == 'Spring':
        data[i][11] = 1
    elif data[i][11] == 'Summer':
        data[i][11] = 2
    elif data[i][11] == 'Autumn':
        data[i][11] = 3

data = data.astype(numpy.float)

# delete all line of data where the value of last column is 0
tmp_data = data
for i in range(0, int(len(data))):
    if data[i][13] == 0:
        tmp_data = numpy.delete(data, i, 0)

data = tmp_data

# keep only the line where the value of column 11 is 4
winter_data = data
for i in range(0, int(len(data))):
    if data[i][11] != 4:
        winter_data = numpy.delete(data, i, 0)


pourcentage = 0.7


def split_data(data, pourcentage):
    numpy.random.shuffle(data)
    split = int(len(data) * pourcentage)
    return data[:split], data[split:]


train, test = split_data(data, pourcentage)
train_winter, test_winter = split_data(winter_data, pourcentage)

# extract second column as target both for train and test
target_train = [i[1] for i in train]
target_test = [i[1] for i in test]

target_train_winter = [i[1] for i in train_winter]
target_test_winter = [i[1] for i in test_winter]

# delete second column from train and test
train = numpy.delete(train, 1, 1)
test = numpy.delete(test, 1, 1)
train_winter = numpy.delete(train_winter, 1, 1)
test_winter = numpy.delete(test_winter, 1, 1)
# delete last column from train and test
train = numpy.delete(train, 11, 1)
test = numpy.delete(test, 11, 1)
train = numpy.delete(train, 11, 1)
test = numpy.delete(test, 11, 1)
train = numpy.delete(train, 7, 1)
test = numpy.delete(test, 7, 1)


reg = LinearRegression().fit(train, target_train)
reg2 = LinearRegression().fit(train_winter, target_train_winter)

y_prediction = reg.predict(test)
print(y_prediction)

print(reg)
print(reg2.score(test_winter, target_test_winter))
