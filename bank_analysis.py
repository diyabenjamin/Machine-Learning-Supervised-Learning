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
import time

# CAR EVALUATION DATASET

# X = data.drop(['id','name'], axis=1)

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
# data_bank['contact']= label_encoder.fit_transform(data_bank['contact'])
# data_bank['month']= label_encoder.fit_transform(data_bank['month'])
data_bank['poutcome']= label_encoder.fit_transform(data_bank['poutcome'])
data_bank['deposit']= label_encoder.fit_transform(data_bank['deposit'])

# example of one hot encoding: cat_df_flights_onehot = cat_df_flights.copy()
# cat_df_flights_onehot = pd.get_dummies(cat_df_flights_onehot, columns=['carrier'], prefix = ['carrier'])
# print(cat_df_flights_onehot.head())

X = data_bank.iloc[:, :-1].values
y = data_bank.iloc[:, -1].values
# print(len(y2[y2==3]))
# 1 = 5289, 0 = 5873 -> 11162 and 0=384, 1=69, 2=1210, 3=65 -> 1728

clf_accuracy = []
training_times = []
testing_times = []



# # Decision Tree
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
# clf = DecisionTreeClassifier(random_state=0)
# clf = clf.fit(x_train, y_train)
# y_pred = clf.predict(x_test)
# print('Accuracy before pruning and HP tuning is %.2f%%' % (accuracy_score(y_test, y_pred) * 100))
# print(classification_report(y_test, y_pred))
#
# depth_range = np.arange(1, 26)
# train_scores, test_scores = validation_curve(clf, x_train, y_train, param_name="max_depth", param_range=depth_range, cv=5)
# plt.figure()
# plt.title('Validation Curve for DT for DataSet 2 - Bank')
# plt.xlabel('max_depth')
# plt.ylabel("Classification score")
# plt.plot(depth_range, np.mean(train_scores, axis=1), label='Training score')
# plt.plot(depth_range, np.mean(test_scores, axis=1), label='Cross-validation score')
# plt.legend(loc="best")
# plt.grid()
# # plt.xticks(depth_range)
# plt.show()
#
# hyperparameters = {'max_depth' : np.arange(20) + 1, 'min_samples_leaf':range(1,101,20)}
# clf_dt = GridSearchCV(clf, param_grid=hyperparameters, cv=5)
# start_time0 = time.time()
# clf_dt.fit(x_train, y_train)
# end_time0 = time.time()
# training_times.append(end_time0 - start_time0)
# print('train time:',end_time0 - start_time0)
# best_dt_params = clf_dt.best_params_
# print(best_dt_params)
# best_max_depth = clf_dt.best_estimator_.get_params()['max_depth']
# best_min_s_leaf = clf_dt.best_estimator_.get_params()['min_samples_leaf']
# start_time1 = time.time()
# y_pred = clf_dt.predict(x_test)
# end_time1 = time.time()
# testing_times.append(end_time1 - start_time1)
# print('test time:',end_time1 - start_time1)
# accuracy_dt = accuracy_score(y_test, y_pred)
# clf_accuracy.append(accuracy_dt)
# print('Accuracy after Tuning is %.2f%%' % (accuracy_dt * 100))
# print(classification_report(y_test, y_pred))
#
# train_sizes = np.linspace(0.1, 1.0, 5)
# cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
# clf = DecisionTreeClassifier(max_depth=best_max_depth, min_samples_leaf= best_min_s_leaf, random_state=0, criterion='gini')
# train_sizes, train_scores, test_scores = \
#         learning_curve(clf, x_train, y_train, train_sizes=train_sizes, cv=cv, n_jobs=4)
# plt.figure()
# plt.title("Learning Curve for DT for Dataset 2 - Bank")
# plt.xlabel('Training sizes')
# plt.ylabel("Classification score")
# plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="r", label="Training score")
# plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color="g", label="Cross-validation score")
# plt.legend(loc="best")
# plt.grid()
# plt.show()


# https://www.dataquest.io/blog/learning-curves-machine-learning/
# car - 1728 -> 80%: 1382 -> 20%: 346
# train_sizes1 = [1, 50, 100, 500, 1000, 1382]

# bank - 11162 -> 80%:8929 -> 20%: 2232
# train_sizes2 = [1, 100, 500, 2000, 5000, 8929]




# Neural Network

# num_hid_layers = np.linspace(1, 150, 10)

# clf = MLPClassifier(hidden_layer_sizes=(10,2), solver='sgd', activation='logistic',
#                        learning_rate_init=0.05, random_state=0) #100

# clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(20, 5), random_state=0, activation='logistic')
#

# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# clf = MLPClassifier(solver='sgd', activation='logistic',
#                         learning_rate_init=0.05, random_state=0) #100
# clf.fit(x_train, y_train)
# y_pred = clf.predict(x_test)
# print(classification_report(y_test, y_pred))
# nn_accuracy = accuracy_score(y_test, y_pred)
# print('Accuracy of neural network is %.2f%%' % (nn_accuracy * 100))
#
# # num_hid_layers = np.linspace(1, 150, 10)
# learning_rate = np.linspace(0.001, 0.2, 5)
# train_scores, test_scores = validation_curve(clf, x_train, y_train, param_name="learning_rate_init", param_range=learning_rate, cv=5)
# # print('Scores: ',train_scores, test_scores)
# plt.figure()
# plt.title('Validation Curve for NN for DataSet 2 - Bank')
# plt.xlabel('Learning Rates')
# plt.ylabel("Classification score")
# plt.plot(learning_rate, np.mean(train_scores, axis=1), label='Training score')
# plt.plot(learning_rate, np.mean(test_scores, axis=1), label='Cross-validation score')
# plt.legend(loc="best")
# plt.grid()
# # plt.xticks(num_hid_layers)
# plt.show()
#
# h_units = [5, 10, 20, 30, 40, 50, 75, 100]
# learning_rates = [0.01, 0.05, 0.1]
# hyperparameters = {'hidden_layer_sizes': h_units, 'learning_rate_init': learning_rates}
# clf_nn = GridSearchCV(MLPClassifier(), hyperparameters, refit=True, verbose=3)
# start_time0 = time.time()
# clf_nn.fit(x_train, y_train)
# end_time0 = time.time()
# training_times.append(end_time0 - start_time0)
# print('train time:',end_time0 - start_time0)
# #Print The value of best Hyperparameters
# # clf_nn.best_params_['hidden_layer_sizes']
# best_h_layer_sizes = clf_nn.best_estimator_.get_params()['hidden_layer_sizes']
# best_learn_rate = clf_nn.best_estimator_.get_params()['learning_rate_init']
# print('Best hidden layer size:', best_h_layer_sizes)
# print('Best learning rate:', best_learn_rate)
#
# start_time1 = time.time()
# y_pred = clf_nn.predict(x_test)
# end_time1 = time.time()
# testing_times.append(end_time1 - start_time1)
# print('test time:',end_time1 - start_time1)
# print(classification_report(y_test, y_pred))
# nn_accuracy = accuracy_score(y_test, y_pred)
# clf_accuracy.append(nn_accuracy)
# print('Accuracy of SVM is %.2f%%' % (nn_accuracy * 100))
#
# # Learning curve - NN
# train_sizes = np.linspace(0.1, 1.0, 5)
# best_clf_nn = MLPClassifier(solver='sgd', activation='logistic', hidden_layer_sizes = best_h_layer_sizes,
#                         learning_rate_init=best_learn_rate, random_state=0) #100
# _, train_scores, test_scores = learning_curve(best_clf_nn, x_train, y_train, train_sizes=train_sizes, cv=5)
# plt.figure()
# plt.title('Learning curve for NN for Dataset 2 - Bank')
# plt.xlabel('Training examples')
# plt.ylabel("Classification score")
# plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
# plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
# plt.legend(loc="best")
# plt.grid()
# plt.show()



# # Boosting
# # K_fold = StratifiedKFold(n_splits=10)
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# adaboost_dt = DecisionTreeClassifier(max_depth=3)
# clf_adaboost = AdaBoostClassifier(base_estimator=adaboost_dt, n_estimators=100, random_state=4)
# clf_adaboost.fit(x_train, y_train)
# y_pred = clf_adaboost.predict(x_test)
# boost_accuracy = accuracy_score(y_test, y_pred)
# print('Accuracy of Boosting is %.2f%%' % (boost_accuracy * 100))
# print(classification_report(y_test, y_pred))
#
# # model complexity analysis
# n_learners = np.arange(200)+1
# train_scores, test_scores = validation_curve(clf_adaboost, x_train, y_train, param_name='n_estimators', param_range=n_learners, cv=5)
# plt.figure()
# plt.title('Validation Curve for AdaBoost DT for DataSet 2 - Bank')
# plt.xlabel('Number of weak learners')
# plt.ylabel("Classification score")
# plt.plot(n_learners, np.mean(train_scores, axis=1), label='Training score')
# plt.plot(n_learners, np.mean(test_scores, axis=1), label='Cross-validation score')
# plt.legend(loc="best")
# plt.grid()
# # plt.xticks(n_estimators_range)
# plt.show()
#
# # BoostedDT HP tuning
# hyperparameters = {'n_estimators':np.linspace(10,100,3).round().astype('int'),
#               'learning_rate': np.linspace(.001,.1,3)}
# grid_boost = GridSearchCV(estimator = AdaBoostClassifier(base_estimator=adaboost_dt),\
#                           param_grid = hyperparameters, scoring='accuracy', n_jobs=4, cv=5)
# start_time0 = time.time()
# grid_boost.fit(x_train, y_train)
# end_time0 = time.time()
# training_times.append(end_time0 - start_time0)
# print('training time: ',end_time0 - start_time0)
# n_estimators = grid_boost.best_params_['n_estimators']
# learning_rate = grid_boost.best_params_['learning_rate']
# print('Best n_estimators:', grid_boost.best_params_['n_estimators'])
# print('Best learning_rate:', grid_boost.best_params_['learning_rate'])
#
# start_time1 = time.time()
# y_pred = grid_boost.predict(x_test)
# end_time1 = time.time()
# testing_times.append(end_time1 - start_time1)
# print('test time: ', end_time1 - start_time1)
# print('Accuracy after Tuning is %.2f%%' % (accuracy_score(y_test, y_pred) * 100))
# print(classification_report(y_test, y_pred))
#
# # learning curve analysis
# clf = AdaBoostClassifier(base_estimator=adaboost_dt, learning_rate=learning_rate, n_estimators=n_estimators, random_state=4)
# train_sizes = np.linspace(0.1, 1.0, 5)
# _, train_scores, test_scores = learning_curve(clf, x_train, y_train, train_sizes=train_sizes, cv=5)
# plt.figure()
# plt.title('Learning curve for AdaBoost DT for Dataset 2 - Bank')
# plt.xlabel('Training examples')
# plt.ylabel("Classification score")
# plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
# plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
# plt.legend(loc="best")
# plt.show()




# SVM

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
clf = svm.SVC()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))
print('Accuracy of SVM is %.2f%%' % (accuracy_score(y_test, y_pred) * 100))

# Model complexity curve
kernels = ['linear', 'rbf']
train_scores, test_scores = validation_curve(clf, x_train, y_train, param_name="kernel", param_range=kernels, n_jobs=1)
plt.figure()
plt.title('Validation Curve for SVM for DataSet 2 - Bank')
plt.xlabel('Kernel functions')
plt.ylabel("Classification score")
plt.plot(kernels, np.mean(train_scores, axis=1), label='Training score')
plt.plot(kernels, np.mean(test_scores, axis=1), label='Cross-validation score')
plt.legend(loc="best")
plt.grid()
plt.show()


hyperparameters = {'C': np.logspace(-3, 2, 6),
              'gamma': [0.1, 1, 10, 100]}
clf_svm = GridSearchCV(svm.SVC(kernel='linear'), hyperparameters, cv=5)
start_time0 = time.time()
clf_svm.fit(x_train, y_train)
end_time0 = time.time()
training_times.append(end_time0 - start_time0)
print('train time:',end_time0 - start_time0)
#Print The value of best Hyperparameters
best_c = clf_svm.best_estimator_.get_params()['C']
best_gamma = clf_svm.best_estimator_.get_params()['gamma']
print('Best C:', best_c)
print('Best gamma:', best_gamma)

# print(clf_svm.best_params_)
# print(clf_svm.best_estimator_)
# https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/

start_time1 = time.time()
y_pred = clf_svm.predict(x_test)
end_time1 = time.time()
testing_times.append(end_time1 - start_time1)
print('test time:',end_time1 - start_time1)
print(classification_report(y_test, y_pred))
svm_accuracy = accuracy_score(y_test, y_pred)
clf_accuracy.append(svm_accuracy)
print('Accuracy of SVM is %.2f%%' % (svm_accuracy * 100))

# Learning curve SVM
train_sizes = np.linspace(0.1, 1.0, 5)
best_clf_svm = svm.SVC(gamma=best_gamma, C=best_c, kernel='linear')
_, train_scores, test_scores = learning_curve(best_clf_svm, x_train, y_train, train_sizes=train_sizes, cv=5)
plt.figure()
plt.title('Learning curve for SVM for Dataset 2 - Bank')
plt.xlabel('Training examples')
plt.ylabel("Classification score")
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
plt.legend(loc="best")
plt.grid()
plt.show()


#kNN
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
#
# #Create KNN Object.
# knn = KNeighborsClassifier()
# #Training the model.
# knn.fit(x_train, y_train)
# #Predict test data set.
# y_pred = knn.predict(x_test)
# #Checking performance of model with classification report.
# print(classification_report(y_test, y_pred))
#
# # Model complexity analysis
# K_fold = StratifiedKFold(10)
# param_range = np.arange(1, 51)
# train_scores, test_scores = validation_curve(knn, x_train, y_train, param_name="n_neighbors",
#     param_range=param_range, cv=K_fold)
# plt.figure()
# plt.title('Validation Curve for kNN for DataSet 2 - Bank')
# plt.xlabel('k')
# plt.ylabel("Classification score")
# plt.plot(param_range, np.mean(train_scores, axis=1), label='Training score')
# plt.plot(param_range, np.mean(test_scores, axis=1), label='Cross-validation score')
# plt.legend(loc="best")
# plt.grid()
# plt.show()
#
# #List of Hyperparameters to tune.
# leaf_size = list(range(1,50))
# n_neighbors = list(range(1,30))
# p=[1,2]
# # hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
# #Use GridSearch
# hyperparameters = {'leaf_size' : leaf_size, 'n_neighbors':n_neighbors, 'p':p}
# clf = GridSearchCV(KNeighborsClassifier(), param_grid=hyperparameters, cv=K_fold)
# #Fit the model
# start_time0 = time.time()
# best_model = clf.fit(x_train,y_train)
# end_time0 = time.time()
# training_times.append(end_time0 - start_time0)
# print(end_time0 - start_time0)
# #Print The value of best Hyperparameters
# print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
# print('Best p:', best_model.best_estimator_.get_params()['p'])
# print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
#
# #Predict test data set.
# start_time1 = time.time()
# y_pred = clf.predict(x_test)
# end_time1 = time.time()
# testing_times.append(end_time1 - start_time1)
# print(end_time1 - start_time1)
# #Checking performance our model with classification report.
# print(classification_report(y_test, y_pred))
# clf_accuracy.append(accuracy_score(y_test, y_pred))
#
# # Learning curve analysis
# train_sizes = np.linspace(0.1, 1.0, 5)
# best_clf = KNeighborsClassifier(n_neighbors=23, leaf_size=1, p=1)
# _, train_scores, test_scores = learning_curve(best_clf, x_train, y_train, train_sizes=train_sizes, cv=K_fold)
# plt.figure()
# plt.title('Learning curve for kNN for Dataset 2 - Bank')
# plt.xlabel('Training examples')
# plt.ylabel("Classification score")
# plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
# plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
# plt.legend(loc="best")
# plt.grid()
# plt.show()



# Comparison of classifiers

# # kNN - training time = 1663.5918221473694 and testing time = 0.2460789680480957

# classifiers = ('Decision tree', 'Neural Network', 'GradientBoosting', 'SVM', 'kNN')
# y_pos = np.arange(len(classifiers))

# plt.figure()
# plt.barh(y_pos, clf_accuracy)
# plt.gca().invert_yaxis()  # labels read top-to-bottom
# plt.yticks(y_pos, classifiers)
# plt.title('Comparison of accuracy')
# plt.xlabel('Accuracy')
# plt.show()
#
# plt.figure()
# plt.barh(y_pos, training_times)
# plt.gca().invert_yaxis()  # labels read top-to-bottom
# plt.yticks(y_pos, classifiers)
# plt.title('Comparison of training time')
# plt.xlabel('Training time (in seconds)')
# plt.show()
#
# plt.figure()
# plt.barh(y_pos, testing_times)
# plt.gca().invert_yaxis()  # labels read top-to-bottom
# plt.yticks(y_pos, classifiers)
# plt.title('Comparison of testing time')
# plt.xlabel('Testing time (in seconds)')
# plt.show()