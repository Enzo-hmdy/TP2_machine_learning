
from distutils.util import copydir_run_2to3
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

import csv
import numpy
import sklearn
from datetime import *
import os
import glob
import csv
import seaborn as sns
import pandas as pd
# Chargement des données
data = csv.reader(open("SeoulBikedata.csv", "r"),
                  delimiter=",")

data = list(data)

# convert data into dataframe


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
# delete 7th column
data = numpy.delete(data, 7, 1)


def hour(h):
    if h >= 17 and h <= 22:
        return 1
    elif h >= 7 and h <= 10:
        return 2
    elif h >= 11 and h <= 16:
        return 3
    else:
        return 4


for i in range(0, int(len(data))):
    data[i][2] = hour(data[i][2])


def plot_correlation_matrix(data):
    corr = data.corr()
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
    plt.show()
    print(corr)


plot_correlation_matrix(pd.DataFrame(data))


figure, axis = plt.subplots(2, 2)

# For Sine Function
axis[0, 0].plot(data[:, 3], data[:, 1], '.')
axis[0, 0].set_title("Temperature vs Count")

axis[0, 1].plot(data[:, 9], data[:, 1], '.')
axis[0, 1].set_title("Pluie vs Count")

axis[1, 0].plot(data[:, 10], data[:, 1], '.')
axis[1, 0].set_title("Neige vs Count")

axis[1, 1].plot(data[:, 5], data[:, 1], '.')
axis[1, 1].set_title("Vent vs Count")


plt.show()


# On sépare les données en données d'entrainement et données de test


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

"""X_data = numpy.delete(data, 1, 1)
Y_data = data[:, 1]

X_winter_data = numpy.delete(winter_data, 1, 1)
Y_winter_data = winter_data[:, 1]

X_spring_data = numpy.delete(spring_data, 1, 1)
Y_spring_data = spring_data[:, 1]

X_summer_data = numpy.delete(summer_data, 1, 1)
Y_summer_data = summer_data[:, 1]

X_autumn_data = numpy.delete(autumn_data, 1, 1)
Y_autumn_data = autumn_data[:, 1]


# Train test split our data
""train, test, target_train, target_test = train_test_split(
    X_data, Y_data, test_size=0.999, random_state=100)
train_winter, test_winter, target_train_winter, target_test_winter = train_test_split(
    X_winter_data, Y_winter_data, test_size=0.25, random_state=100)
train_spring, test_spring, target_train_spring, target_test_spbring = train_test_split(
    X_spring_data, Y_spring_data, test_size=0.25, random_state=100)
train_summer, test_summer, target_train_summer, target_test_summer = train_test_split(
    X_summer_data, Y_summer_data, test_size=0.25, random_state=100)
train_autumn, test_autumn, target_train_autumn, target_test_autumn = train_test_split(
    X_autumn_data, Y_autumn_data, test_size=0.25, random_state=100)"""

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


# transpose all target train and test
target_train = target_train
target_test = target_test

target_train_winter = target_train_winter
target_test_winter = target_test_winter

target_train_spring = target_train_spring
target_test_spring = target_test_spring

target_train_summer = target_train_summer
target_test_summer = target_test_summer

target_train_autumn = target_train_autumn
target_test_autumn = target_test_autumn


# Pour tous ls train et test on supprime la colonne qui contient le nombre de vélos loués
all_test_train = [train, test, train_winter, test_winter, train_spring,
                  test_spring, train_summer, test_summer, train_autumn, test_autumn]
for i in range(0, int(len(all_test_train))):
    all_test_train[i] = numpy.delete(all_test_train[i], 1, 1)
train, test, train_winter, test_winter, train_spring, test_spring, train_summer, test_summer, train_autumn, test_autumn = all_test_train


# On réalise la regression sur chaque set
# print shape of train and target train
reg_total = LinearRegression().fit(train, target_train)
reg_winter = LinearRegression().fit(train_winter, target_train_winter)
reg_spring = LinearRegression().fit(train_spring, target_train_spring)
reg_summer = LinearRegression().fit(train_summer, target_train_summer)
reg_autumn = LinearRegression().fit(train_autumn, target_train_autumn)

# Regression lasso
reg_lasso_total = linear_model.Lasso(alpha=0.01).fit(train, target_train)
reg_lasso_winter = linear_model.Lasso(
    alpha=0.01).fit(train_winter, target_train_winter)
reg_lasso_spring = linear_model.Lasso(
    alpha=0.01).fit(train_spring, target_train_spring)
reg_lasso_summer = linear_model.Lasso(
    alpha=0.01).fit(train_summer, target_train_summer)
reg_lasso_autumn = linear_model.Lasso(
    alpha=0.01).fit(train_autumn, target_train_autumn)

# Regression ridge
reg_ridge_total = linear_model.Ridge(alpha=0.01).fit(train, target_train)
reg_ridge_winter = linear_model.Ridge(
    alpha=0.01).fit(train_winter, target_train_winter)
reg_ridge_spring = linear_model.Ridge(
    alpha=0.01).fit(train_spring, target_train_spring)
reg_ridge_summer = linear_model.Ridge(
    alpha=0.01).fit(train_summer, target_train_summer)
reg_ridge_autumn = linear_model.Ridge(
    alpha=0.01).fit(train_autumn, target_train_autumn)


all_reg = [reg_total, reg_winter, reg_spring, reg_summer, reg_autumn]
all_laso_reg = [reg_lasso_total, reg_lasso_winter,
                reg_lasso_spring, reg_lasso_summer, reg_lasso_autumn]
all_ridge_reg = [reg_ridge_total, reg_ridge_winter,
                 reg_lasso_spring, reg_lasso_summer, reg_lasso_autumn]


all_test = [test, test_winter, test_spring, test_summer, test_autumn]
all_train = [train, train_winter, train_spring, train_summer, train_autumn]

all_target_test = [target_test, target_test_winter,
                   target_test_spring, target_test_summer, target_test_autumn]
all_target_train = [target_train, target_train_winter,
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


def get_MSE(target_test, test):
    score = mean_squared_error(target_test, test)
    # print the name of variable the regression and the score
    print("MEAN SQARED ERROR : ", score)


print("\n\n ----------NORMAL -----------------------------\n\n")
for i in range(0, int(len(all_reg))):
    get_score(all_reg[i], all_test[i], all_target_test[i])
print("\n\n ----------LASSO -----------------------------\n\n")
for i in range(0, int(len(all_laso_reg))):
    get_score(all_laso_reg[i], all_test[i], all_target_test[i])

print("\n\n ----------RIDGE-------------------\n\n")
for i in range(0, int(len(all_ridge_reg))):

    get_score(all_ridge_reg[i], all_test[i], all_target_test[i])

print("\n\n ---------- MSE -----------------------------\n\n")
for i in range(0, int(len(all_reg))):
    get_MSE(all_target_test[i], all_reg[i].predict(all_test[i]))


Y_pred = reg_total.predict(test)


# ploting the line graph of actual and predicted values
plt.figure(figsize=(12, 5))
plt.plot((Y_pred)[:80])
plt.plot((np.array(target_test)[:80]))
plt.legend(["Prediction", "valeur réelle"])
plt.show()


param_grid = {"n_estimators":[50,100,150],
              'max_depth' : [10,15,20,25,'none'],
              'min_samples_split': [10,50,100],
              'max_features' :[24,35,40,49]}

model1 = RandomForestRegressor()
grid1 = GridSearchCV(estimator=model1, param_grid=param_grid)
grid1.fit(train, target_train)

# Regression random forest
reg_forest_total = RandomForestRegressor(n_estimators = 100, random_state = 0).fit(train, target_train)
reg_forest_winter = RandomForestRegressor(n_estimators = 100, random_state = 0).fit(train_winter, target_train_winter)
reg_forest_spring = RandomForestRegressor(n_estimators = 100, random_state = 0).fit(train_spring, target_train_spring)
reg_forest_summer = RandomForestRegressor(n_estimators = 100, random_state = 0).fit(train_summer, target_train_summer)
reg_forest_autumn = RandomForestRegressor(n_estimators = 100, random_state = 0).fit(train_autumn, target_train_autumn)

all_forest_reg = [reg_forest_total, reg_forest_winter, reg_forest_spring, reg_forest_summer, reg_forest_autumn]
print("\n\n ---------- NORMAL DECISION TREE -----------------------------\n\n")
for i in range(0, int(len(all_forest_reg))):
    get_score(all_forest_reg[i], all_test[i], all_target_test[i])


Y_forest_pred = all_forest_reg.predict(test)

# ploting the line graph of actual and predicted values
plt.figure(figsize=(12, 5))
plt.plot((Y_forest_pred)[:80])
plt.plot((np.array(target_test)[:80]))
plt.legend(["Prediction", "valeur réelle"])
plt.show()

