
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


pourcentage = 0.7


def split_data(data, pourcentage):
    numpy.random.shuffle(data)
    split = int(len(data) * pourcentage)
    return data[:split], data[split:]


train, test = split_data(data, pourcentage)

# extract second column as target both for train and test
target_train = [i[1] for i in train]
target_test = [i[1] for i in test]

# delete second column from train and test
train = numpy.delete(train, 1, 1)
test = numpy.delete(test, 1, 1)


reg = LinearRegression().fit(train, target_train)

"""def mse(coef, x, y):
    return np.mean((y - np.dot(x, coef))**2)


def gradients(coef, x, y):
    return -2 * np.dot(x.T, y - np.dot(x, coef)) / y.size


def multilinear_regression(coef, x, y, lr, b1=0.9, b2=0.999, epsilon=1e-8):
    prev_error = 0
    m_coef = np.zeros(coef.shape)
    v_coef = np.zeros(coef.shape)
    moment_m_coef = np.zeros(coef.shape)
    moment_v_coef = np.zeros(coef.shape)
    t = 0

    while True:
        error = mse(coef, x, y)
        if abs(error - prev_error) <= epsilon:
            break
        prev_error = error
        grad = gradients(coef, x, y)
        t += 1
        m_coef = b1 * m_coef + (1-b1)*grad
        v_coef = b2 * v_coef + (1-b2)*grad**2
        moment_m_coef = m_coef / (1-b1**t)
        moment_v_coef = v_coef / (1-b2**t)

        delta = ((lr / moment_v_coef**0.5 + 1e-8) *
                 (b1 * moment_m_coef + (1-b1)*grad/(1-b1**t)))

        coef = np.subtract(coef, delta)
    return coef"""
