
from distutils.util import copydir_run_2to3
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
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

# on transforme notre colonne de dates en colonne de jours de de la semaine 0 à 6 (0 = lundi, 6 = dimanche)


def date_to_days(tab_date):
    day = numpy.zeros(int(len(tab_date)))
    for i in range(0, int(len(tab_date))):
        day[i] = tab_date[i].weekday()
    return day


dates = date_to_days(dates)
for i in range(0, int(len(dates))):
    data[i][0] = dates[i]

# On gère les données qui ne sont pas des nombres
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

# On supprime les données dont la valeur de la dernière colonne est 0 car cela signifie que service n'était pas en marche
# Donc que les données ne sont pas pertinentes
tmp_data = data
for i in range(0, int(len(data))):
    if data[i][13] == 0:
        tmp_data = numpy.delete(data, i, 0)

data = tmp_data


# On garde les données correspondant à l'hiver
winter_data = data
for i in range(0, int(len(data))):
    if data[i][11] != 4:
        winter_data = numpy.delete(data, i, 0)

# on garde les données correspondant au printemps
spring_data = data
for i in range(0, int(len(data))):
    if data[i][11] != 1:
        spring_data = numpy.delete(data, i, 0)

# on garde les données correspondant à l'été
summer_data = data
for i in range(0, int(len(data))):
    if data[i][11] != 2:
        summer_data = numpy.delete(data, i, 0)

# on garde les données correspondant à l'automne
autumn_data = data
for i in range(0, int(len(data))):
    if data[i][11] != 3:
        autumn_data = numpy.delete(data, i, 0)


pourcentage = 0.7

# On sépare les données en données d'entrainement et données de test


def split_data(data, pourcentage):
    numpy.random.shuffle(data)
    split = int(len(data) * pourcentage)
    return data[:split], data[split:]


# On récupère les données d'entrainement et de test
train, test = split_data(data, pourcentage)
train_winter, test_winter = split_data(winter_data, pourcentage)
train_spring, test_spring = split_data(spring_data, pourcentage)
train_summer, test_summer = split_data(summer_data, pourcentage)
train_autumn, test_autumn = split_data(autumn_data, pourcentage)


# On extrait les données qui serviront d'objectif à atteindre, soit ici le nombre de vélos loués
target_train = [i[1] for i in train]
target_test = [i[1] for i in test]

target_train_winter = [i[1] for i in train_winter]
target_test_winter = [i[1] for i in test_winter]

target_train_spring = [i[1] for i in train_spring]
target_test_spring = [i[1] for i in test_spring]

target_train_summer = [i[1] for i in train_summer]
target_test_summer = [i[1] for i in test_summer]

target_train_autumn = [i[1] for i in train_autumn]
target_test_autumn = [i[1] for i in test_autumn]


# Pour tous ls train et test on supprime la colonne qui contient le nombre de vélos loués
all_test_train = [train, test, train_winter, test_winter, train_spring,
                  test_spring, train_summer, test_summer, train_autumn, test_autumn]
for i in range(0, int(len(all_test_train))):
    all_test_train[i] = numpy.delete(all_test_train[i], 1, 1)
train, test, train_winter, test_winter, train_spring, test_spring, train_summer, test_summer, train_autumn, test_autumn = all_test_train


# On réalise la regression sur chaque set
reg_total = LinearRegression().fit(train, target_train)
reg_winter = LinearRegression().fit(train_winter, target_train_winter)
reg_spring = LinearRegression().fit(train_spring, target_train_spring)
reg_summer = LinearRegression().fit(train_summer, target_train_summer)
reg_autumn = LinearRegression().fit(train_autumn, target_train_autumn)

all_reg = [reg_total, reg_winter, reg_spring, reg_summer, reg_autumn]
all_test = [test, test_winter, test_spring, test_summer, test_autumn]
all_target = [target_test, target_test_winter,
              target_test_spring, target_test_summer, target_test_autumn]

# On récupère le score de chaque regression


def get_score(reg, test, target_test):
    score = reg.score(test, target_test)
    # print the name of variable the regression and the score
    print("Score de la regression : ", score)


def get_mean_score(reg, test, target_test):
    score = 0
    for i in range(0, 10000):
        score += reg.score(test, target_test)
    print("mean score : ", score / 10000)


for i in range(0, int(len(all_reg))):
    get_score(all_reg[i], all_test[i], all_target[i])
    #get_mean_score(all_reg[i], all_test[i], all_target[i])
