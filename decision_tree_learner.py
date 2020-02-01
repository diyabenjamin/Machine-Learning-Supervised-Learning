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

# print('y=', (y[y == 2].shape[0]/y.shape[0]*100.0))

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

# example of one hot encoding: cat_df_flights_onehot = cat_df_flights.copy()
# cat_df_flights_onehot = pd.get_dummies(cat_df_flights_onehot, columns=['carrier'], prefix = ['carrier'])
# print(cat_df_flights_onehot.head())

X = data_bank.iloc[:, :-1].values
y = data_bank.iloc[:, -1].values
# print(len(y2[y2==3]))
# 1 = 5289, 0 = 5873 -> 11162 and 0=384, 1=69, 2=1210, 3=65 -> 1728


# Decision Tree
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print('Accuracy is %.2f%%' % (accuracy_score(y_test, y_pred) * 100))

depth_range = np.arange(1, 26)
train_scores, test_scores = validation_curve(clf, x_train, y_train, param_name="max_depth", param_range=depth_range, cv=5)
plt.figure()
plt.title('Validation Curve for DT for DataSet 1 - car')
plt.xlabel('max_depth')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
# plt.xticks(depth_range)
plt.plot(depth_range, np.mean(train_scores, axis=1), label='Training score')
plt.plot(depth_range, np.mean(test_scores, axis=1), label='Cross-validation score')
plt.show()

param_range = {'max_depth' : np.arange(10) + 1, 'min_samples_leaf':range(1,101,20)}
clf_dt = GridSearchCV(clf, param_grid=param_range, cv=5)
# t0 = time.time()
clf_dt.fit(x_train, y_train)
best_dt_params = clf_dt.best_params_
print(best_dt_params)
y_pred = clf_dt.predict(x_test)
print('Accuracy after Tuning is %.2f%%' % (accuracy_score(y_test, y_pred) * 100))

train_sizes = np.linspace(0.1, 1.0, 5)
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(clf_dt, x_train, y_train, train_sizes=train_sizes, cv=cv, n_jobs=4,
                       return_times=True)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
fit_times_mean = np.mean(fit_times, axis=1)
plt.figure()
plt.title("Learning Curve for DT for Dataset 1 - car")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.show()

# https://www.dataquest.io/blog/learning-curves-machine-learning/
# car - 1728 -> 80%: 1382 -> 20%: 346
# train_sizes1 = [1, 50, 100, 500, 1000, 1382]

# bank - 11162 -> 80%:8929 -> 20%: 2232
# train_sizes2 = [1, 100, 500, 2000, 5000, 8929]



# # Neural Network
#
# # do we need random_state below ??????
#
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# f1_test = []
# f1_train = []
# num_hid_layers = np.linspace(1, 150, 10)
#
# clf = MLPClassifier(hidden_layer_sizes=(10,2), solver='sgd', activation='logistic',
#                         learning_rate_init=0.05, random_state=0) #100
# clf.fit(x_train, y_train)
# y_pred = clf.predict(x_test)
# nn_accuracy = accuracy_score(y_test, y_pred)
# print('Accuracy of neural network is %.2f%%' % (nn_accuracy * 100))
#
# train_scores, test_scores = validation_curve(clf, x_train, y_train, param_name="alpha", param_range=num_hid_layers, cv=5)
# # print('Scores: ',train_scores, test_scores)
# plt.figure()
# plt.title('Validation Curve for NN for DataSet 1 - car')
# plt.xlabel('num_hid_layers')
# plt.ylabel("Classification score")
# plt.legend(loc="best")
# plt.grid()
# plt.xticks(num_hid_layers)
# plt.plot(num_hid_layers, np.mean(train_scores, axis=1), label='Training score')
# plt.plot(num_hid_layers, np.mean(test_scores, axis=1), label='Cross-validation score')
# plt.show()
#
# # for i in num_hid_layers:
# #     clf = MLPClassifier(hidden_layer_sizes=i, solver='sgd', activation='logistic',
# #                         learning_rate_init=0.05, random_state=0) #100
# #     clf.fit(x_train, y_train)
# #     y_pred_test = clf.predict(X_test)
# #     y_pred_train = clf.predict(X_train)
# #     f1_test.append(f1_score(y_test, y_pred_test))
# #     f1_train.append(f1_score(y_train, y_pred_train))
# #
# # plt.plot(hlist, f1_test, 'o-', color='r', label='Test F1 Score')
# # plt.plot(hlist, f1_train, 'o-', color='b', label='Train F1 Score')
# # plt.ylabel('Model F1 Score')
# # plt.xlabel('No. Hidden Units')
# #
# # plt.title(title)
# # plt.legend(loc='best')
# # plt.tight_layout()
# # plt.show()
#
#
#
# # list1 = []
# # list2 = []
# # for i in range(1, 95):
# #     clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(20, 5), random_state=0, activation='logistic')
# #
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=1 - i / 100)
# #     clf.fit(X_train, y_train)
# #     train_predict = clf.predict(X_train)
# #     test_predict = clf.predict(X_test)
# #     list1.append(accuracy_score(y_train, train_predict))
# #     list2.append(accuracy_score(y_test, test_predict))
# # plt.plot(range(len(list2)), list2)
# # plt.plot(range(len(list1)), list1)
# # plt.show()



# Boosting
K_fold = StratifiedKFold(n_splits=10)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
num_learners = 2000
clf_boosted = GradientBoostingClassifier(max_depth=3, n_estimators=num_learners, random_state=7)
clf_boosted.fit(x_train, y_train)
y_pred = clf_boosted.predict(x_test)
boost_accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of Boosting is %.2f%%' % (boost_accuracy * 100))

# model complexity analysis
n_estimators_range = np.arange(num_learners) + 1
train_scores, test_scores = validation_curve(clf_boosted, x_train, y_train, param_name="n_estimators", param_range=n_estimators_range, cv=K_fold)
plt.figure()
plt.title('Validation Curve for Adaboost for DataSet 2 - bank')
plt.xlabel('Number of weak learners')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
# plt.xticks(n_estimators_range)
plt.plot(n_estimators_range, np.mean(train_scores, axis=1), label='Training score')
plt.plot(n_estimators_range, np.mean(test_scores, axis=1), label='Cross-validation score')
plt.show()

# BoostedDT HP tuning
param_test = {'max_depth':range(1,16,1), 'min_samples_split':range(50,501,50)}
dt = DecisionTreeClassifier(max_depth=3)
clf_boost = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, random_state=10),
param_grid = param_test, scoring='accuracy', n_jobs=4, cv=K_fold)
clf_boost.grid_scores_, clf_boost.best_params_, clf_boost.best_score_
print(clf_boost.best_params_)
max_depth = clf_boost.best_params_['max_depth']
min_samples_split = clf_boost.best_params_['min_samples_split']
clf_boost.fit(x_train, y_train)
y_pred = clf_boost.predict(x_test)
print('Accuracy after Tuning is %.2f%%' % (accuracy_score(y_test, y_pred) * 100))

# learning curve analysis
clf = GradientBoostingClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                              learning_rate=0.1, n_estimators=60, random_state=100)
clf.fit(x_train, y_train)
train_sizes = np.linspace(0.1, 1.0, 5)
_, train_scores, test_scores = learning_curve(clf_boost, x_train, y_train, train_sizes=train_sizes, cv=K_fold)
plt.figure()
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
plt.title('Learning curve for Boost DT')
plt.xlabel('Training size')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.show()

# y_pred = clf.predict(x_test)
# boost_accuracy = accuracy_score(y_test, y_pred)
# print('Final Accuracy of Boosting is %.2f%%' % (boost_accuracy * 100))

# param_grid = {'min_samples_leaf': np.linspace(start_leaf_n,end_leaf_n,3).round().astype('int'),
#                   'max_depth': np.arange(1,4),
#                   'n_estimators': np.linspace(10,100,3).round().astype('int'),
#                   'learning_rate': np.linspace(.001,.1,3)}
# clf_boost = AdaBoostClassifier(base_estimator=clf, random_state=2, n_estimators=50)
# clf_boost.fit(x_train, y_train)




# # SVM
#
# clf = svm.SVC(kernel='rbf')
# clf.fit(x_train, y_train)
# # predictions = clf.predict(x_test)
# # print(classification_report(y_test, predictions))
#
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
# # Model complexity curve
# gamma_range = np.logspace(0.1, 1, 10)
# train_scores, test_scores = validation_curve(clf, x_train, y_train, param_name="gamma", param_range=gamma_range, n_jobs=1)
# # print('Scores: ',train_scores, test_scores)
# plt.figure()
# plt.title('Validation Curve for SVM for DataSet 1 - car **')
# plt.xlabel('Gamma')
# plt.ylabel("Classification score")
# plt.legend(loc="best")
# plt.grid()
# plt.xticks(gamma_range)
# plt.plot(gamma_range, np.mean(train_scores, axis=1), label='Training score')
# plt.plot(gamma_range, np.mean(test_scores, axis=1), label='Cross-validation score')
# plt.show()
#
# param_range = np.logspace(-2, 2, 10)
# train_scores, test_scores = validation_curve(
#     clf, x_train, y_train, param_name="C", param_range=param_range,
#     scoring="accuracy", n_jobs=1)
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
# plt.title("Validation Curve with SVM")
# plt.xlabel("Gamma")
# plt.ylabel("Score")
# plt.ylim(0.0, 1.1)
# lw = 2
# plt.semilogx(param_range, train_scores_mean, label="Training score",
#              color="darkorange", lw=lw)
# plt.fill_between(param_range, train_scores_mean - train_scores_std,
#                  train_scores_mean + train_scores_std, alpha=0.2,
#                  color="darkorange", lw=lw)
# plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
#              color="navy", lw=lw)
# plt.fill_between(param_range, test_scores_mean - test_scores_std,
#                  test_scores_mean + test_scores_std, alpha=0.2,
#                  color="navy", lw=lw)
# plt.legend(loc="best")
# plt.show()
#
# # param_grid = {'C': np.logspace(-3, 2, 6),
# #               'gamma': [0.1, 1, 10, 100, 1000],
# #               'kernel': ['rbf']}
# # grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3)
# # grid.fit(x_train, y_train)
# # best_clf = grid
# # print(grid.best_params_)
# # print(grid.best_estimator_)
# # # https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
# # grid_predictions = grid.predict(x_test)
# # print(classification_report(y_test, grid_predictions))
# # svm_accuracy = accuracy_score(y_test, y_pred)
# # print('Accuracy of SVM is %.2f%%' % (svm_accuracy * 100))
#
# # Learning curve SVM
# # best_clf or clf
# train_sizes = np.linspace(0.1, 1.0, 5)
# _, train_scores, test_scores = learning_curve(clf, x_train, y_train, train_sizes=train_sizes, cv=5)
# plt.figure()
# plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
# plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
# plt.title('Learning curve for SVM rbf')
# plt.xlabel('Training size')
# plt.ylabel("Classification score")
# plt.legend(loc="best")
# plt.grid()
# #plt.savefig(fig_path + 'dt_learning_curve.png')
# plt.show()


# kNN
# k = [1, 10, 100, 200]
# for i in k:
#     clf = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
#     clf.fit(x_train, y_train)

# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# # cv = StratifiedKFold(4)
# param_range = np.arange(1, 51)
# train_scores, test_scores = validation_curve(KNeighborsClassifier(), x_train, y_train, param_name="n_neighbors",
#     param_range=param_range, cv=5)
# plt.figure()
# plt.title('Validation Curve for kNN for DataSet 1 - car **')
# plt.xlabel('k')
# plt.ylabel("score")
# plt.legend(loc="best")
# # plt.grid()
# plt.plot(param_range, np.mean(train_scores, axis=1), label='Training score')
# plt.plot(param_range, np.mean(test_scores, axis=1), label='Cross-validation score')
# plt.show()
#
# k_optimal = np.argmax(np.mean(test_scores, axis=1)) + 1
# best_clf = KNeighborsClassifier(n_neighbors=k_optimal)
#
# train_sizes = np.linspace(0.1, 1.0, 5)
# _, train_scores, test_scores = learning_curve(best_clf, x_train, y_train, train_sizes=train_sizes, cv=5)
#
# plt.figure()
# plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
# plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
# plt.title('Learning curve for kNN')
# plt.xlabel('Fraction of training examples')
# plt.ylabel("Classification score")
# plt.legend(loc="best")
# # plt.grid()
# #plt.savefig(fig_path + 'dt_learning_curve.png')
# plt.show()