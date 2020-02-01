from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve, cross_validate, validation_curve
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

# CAR EVALUATION DATASET
data_car = pd.read_csv('car_evaluation.csv')
print('rows: ', len(data_car), ' columns: ', len(data_car.columns))

# X = data.drop(['id','name'], axis=1)

label_encoder = LabelEncoder()
data_car['buying price']= label_encoder.fit_transform(data_car['buying price'])
data_car['maintenance cost']= label_encoder.fit_transform(data_car['maintenance cost'])
data_car['number of doors']= label_encoder.fit_transform(data_car['number of doors'])
data_car['number of persons']= label_encoder.fit_transform(data_car['number of persons'])
data_car['lug_boot']= label_encoder.fit_transform(data_car['lug_boot'])
data_car['safety']= label_encoder.fit_transform(data_car['safety'])
data_car['decision']= label_encoder.fit_transform(data_car['decision'])

X2 = data_car.iloc[:, :-1].values
y2 = data_car.iloc[:, -1].values

# BANK MARKETING DATASET
data_bank = pd.read_csv('bank.csv')
print('rows: ', len(data_bank), ' columns: ', len(data_bank.columns))

label_encoder = LabelEncoder()
# label_encoder = OneHotEncoder()
data_bank['job']= label_encoder.fit_transform(data_bank['job'])
data_bank['marital']= label_encoder.fit_transform(data_bank['marital'])
data_bank['education']= label_encoder.fit_transform(data_bank['education'])
data_bank['default']= label_encoder.fit_transform(data_bank['default'])
data_bank['housing']= label_encoder.fit_transform(data_bank['housing'])
data_bank['loan']= label_encoder.fit_transform(data_bank['loan'])
data_bank['contact']= label_encoder.fit_transform(data_bank['contact'])
data_bank['month']= label_encoder.fit_transform(data_bank['month'])
data_bank['poutcome']= label_encoder.fit_transform(data_bank['poutcome'])
data_bank['deposit']= label_encoder.fit_transform(data_bank['deposit'])

X = data_bank.iloc[:, :-1].values
y = data_bank.iloc[:, -1].values

# SVM

clf = svm.SVC(kernel='rbf')
clf.fit(x_train, y_train)
# predictions = clf.predict(x_test)
# print(classification_report(y_test, predictions))

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model complexity curve
gamma_range = np.logspace(0.1, 1, 10)
train_scores, test_scores = validation_curve(clf, x_train, y_train, param_name="gamma", param_range=gamma_range, n_jobs=1)
# print('Scores: ',train_scores, test_scores)
plt.figure()
plt.title('Validation Curve for SVM for DataSet 1 - car **')
plt.xlabel('Gamma')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.xticks(gamma_range)
plt.plot(gamma_range, np.mean(train_scores, axis=1), label='Training score')
plt.plot(gamma_range, np.mean(test_scores, axis=1), label='Cross-validation score')
plt.show()

param_range = np.logspace(-2, 2, 10)
train_scores, test_scores = validation_curve(
    clf, x_train, y_train, param_name="C", param_range=param_range,
    scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.title("Validation Curve with SVM")
plt.xlabel("Gamma")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()

# param_grid = {'C': np.logspace(-3, 2, 6),
#               'gamma': [0.1, 1, 10, 100, 1000],
#               'kernel': ['rbf']}
# grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3)
# grid.fit(x_train, y_train)
# best_clf = grid
# print(grid.best_params_)
# print(grid.best_estimator_)
# # https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
# grid_predictions = grid.predict(x_test)
# print(classification_report(y_test, grid_predictions))
# svm_accuracy = accuracy_score(y_test, y_pred)
# print('Accuracy of SVM is %.2f%%' % (svm_accuracy * 100))

# Learning curve SVM
# best_clf or clf
train_sizes = np.linspace(0.1, 1.0, 5)
_, train_scores, test_scores = learning_curve(clf, x_train, y_train, train_sizes=train_sizes, cv=5)
plt.figure()
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
plt.title('Learning curve for SVM rbf')
plt.xlabel('Training size')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
#plt.savefig(fig_path + 'dt_learning_curve.png')
plt.show()

