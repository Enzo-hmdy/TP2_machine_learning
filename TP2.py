
from distutils.util import copydir_run_2to3
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean
from numpy import std
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
axis[0, 0].set_title("Nombre vélo vs Temperature")

axis[0, 1].plot(data[:, 9], data[:, 1], '.')
axis[0, 1].set_title("Nombre vélo vs Pluie")

axis[1, 0].plot(data[:, 10], data[:, 1], '.')
axis[1, 0].set_title("Nombre vélo vs Neige")

axis[1, 1].plot(data[:, 5], data[:, 1], '.')
axis[1, 1].set_title("Nombre vélo vs vent")


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

# Regression elastic net
reg_elastic_total = linear_model.ElasticNet().fit(train, target_train)
reg_elastic_winter = linear_model.ElasticNet().fit(
    train_winter, target_train_winter)
reg_elastic_spring = linear_model.ElasticNet().fit(
    train_spring, target_train_spring)
reg_elastic_summer = linear_model.ElasticNet().fit(
    train_summer, target_train_summer)
reg_elastic_autumn = linear_model.ElasticNet().fit(
    train_autumn, target_train_autumn)

# -------------------------------------- GRID SEARCH POUR MEILLEUR PARAMETRE --------------------------------------

# Cette partie était pour tester les meilleurs alpha de la regression ridge et lasso mais le temps d'éxécution était trop long
# donc on a mis en commentaire
#alphas = numpy.arange(0.01, 100, 0.01)
"""model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
grid.fit(train, target_train)
# plot the results with matplotlib histogram separately with log scale for x axis
plt.figure(figsize=(10, 10))
plt.subplot(211)
plt.title('Ridge Regression')
plt.xlabel('alpha')
plt.ylabel('score')
plt.plot(alphas, grid.cv_results_['mean_test_score'])
plt.xscale('log')
plt.show()
# print best alpha
print(grid.best_estimator_.alpha)
"""
"""
model = Lasso()
grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
grid.fit(train, target_train)
# plot the results with matplotlib histogram separately with log scale for x axis
plt.figure(figsize=(10, 10))
plt.subplot(211)
plt.title('Lasso Regression')
plt.xlabel('alpha')
plt.ylabel('score')
plt.plot(alphas, grid.cv_results_['mean_test_score'])
plt.xscale('log')
plt.show()
print("best score ", grid.best_estimator_.alpha)
"""

# gridsearch tunning for elastic net and give best parameters for L1 and alpha


all_reg = [reg_total, reg_winter, reg_spring, reg_summer, reg_autumn]
all_laso_reg = [reg_lasso_total, reg_lasso_winter,
                reg_lasso_spring, reg_lasso_summer, reg_lasso_autumn]
all_ridge_reg = [reg_ridge_total, reg_ridge_winter,
                 reg_lasso_spring, reg_lasso_summer, reg_lasso_autumn]
all_elastic_reg = [reg_elastic_total, reg_elastic_winter,
                   reg_elastic_spring, reg_elastic_summer, reg_elastic_autumn]


all_test = [test, test, test_winter, test_spring, test_summer, test_autumn]
all_train = [train, train_winter, train_spring, train_summer, train_autumn]

all_target_test = [target_test, target_test, target_test_winter,
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

print("\n\n ----------ELASTIC NET-------------------\n\n")
for i in range(0, int(len(all_elastic_reg))):
    get_score(all_elastic_reg[i], all_test[i], all_target_test[i])


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


param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
rf = RandomForestRegressor()
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           n_jobs=-1, verbose=2)
grid_search.fit(train, target_train)
print(grid_search.best_params_)
print(grid_search.best_score_)


# Regression random forest
reg_test = RandomForestRegressor(bootstrap=True, max_depth=20, max_features=24,
                                 min_samples_leaf=3, min_samples_split=10, n_estimators=150).fit(train, target_train)
reg_forest_total = RandomForestRegressor(
    n_estimators=100, random_state=0).fit(train, target_train)
reg_forest_winter = RandomForestRegressor(
    n_estimators=100, random_state=0).fit(train_winter, target_train_winter)
reg_forest_spring = RandomForestRegressor(
    n_estimators=100, random_state=0).fit(train_spring, target_train_spring)
reg_forest_summer = RandomForestRegressor(
    n_estimators=100, random_state=0).fit(train_summer, target_train_summer)
reg_forest_autumn = RandomForestRegressor(
    n_estimators=100, random_state=0).fit(train_autumn, target_train_autumn)

# ----------------------Cross validation----------------------

X_data = numpy.delete(data, 1, 1)
Y_data = data[:, 1]
reg_model = RandomForestRegressor(n_estimators=100, random_state=0)


def get_cross_val_score(model, X_data, Y_data, ka_fold):
    score = cross_val_score(model, X_data, Y_data, cv=ka_fold)
    return mean(score), score.min(), score.max()


def define_best_fold(X_data, Y_data, nbr_fold_max, model):
    moyenne, min, max = list(), list(), list()
    nbr_fold = range(2, nbr_fold_max)
    for i in nbr_fold:
        cv = KFold(n_splits=i, random_state=0, shuffle=True)
        k_moyenne, k_min, k_max = get_cross_val_score(
            model, X_data, Y_data, cv)
        print('> folds=%d, accuracy=%.3f (%.3f,%.3f)' %
              (i, k_moyenne, k_min, k_max))
        moyenne.append(k_moyenne)
        min.append(k_moyenne - k_min)
        max.append(k_max - k_moyenne)
    return moyenne, min, max, nbr_fold


heorical_moyenne, theorical_min, theorical_max = get_cross_val_score(
    reg_model, X_data, Y_data, KFold())


moyenne_total, min_total, max_total, nbr_fold_total = define_best_fold(
    X_data, Y_data, 35, reg_model)

plt.errorbar(nbr_fold_total, moyenne_total, yerr=[
             min_total, max_total], fmt='o', color='red', ecolor='blue', elinewidth=1, capsize=5)

plt.title('Cross validation')
plt.xlabel('Nombre de fold')
plt.ylabel('Score R2')

plt.legend(['Score du modèle en bleu, min et max de la rotation'])
plt.show()


# ------------------------------------------------------------------------

all_forest_reg = [reg_test, reg_forest_total, reg_forest_winter,
                  reg_forest_spring, reg_forest_summer, reg_forest_autumn]
print("\n\n ---------- NORMAL DECISION TREE -----------------------------\n\n")
for i in range(0, int(len(all_forest_reg))):
    get_score(all_forest_reg[i], all_test[i], all_target_test[i])


Y_forest_pred = reg_forest_total.predict(test)

# ploting the line graph of actual and predicted values
plt.figure(figsize=(12, 5))
plt.plot((Y_forest_pred)[:80])
plt.plot((Y_pred)[:80])
plt.plot((np.array(target_test)[:80]))
plt.legend(
    ["Prediction_forect", "Prediction Regression Linéraire", "valeur réelle"])
plt.show()
