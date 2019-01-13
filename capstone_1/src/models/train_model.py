# -*- coding: utf-8 -*-

#
# Functionality to train all models in this project.
#

import logging
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from src.models.data_manager import DataManager
from src.models.rntn import RNTN

#
# Function to train model with predefined split.
# Expects both the training and cross validation dataset.
# Hyper-parameter training done with both.
#


def train_model(clf, params, score_func, x_train, y_train, x_dev, y_dev):
    """Train a given model with the training/cross validation data provided."""

    # Prepare data for training
    validation_set_indexes = [-1] * len(x_train) + [0] * len(x_dev)
    cv = PredefinedSplit(test_fold=validation_set_indexes)

    # Find the best hyper-parameter using GridSearchCV
    model = GridSearchCV(clf, params, scoring=score_func, cv=cv)

    x = np.asarray(x_train + x_dev).reshape(-1, 1)
    y = y_train + y_dev
    model.fit(x, y)

    return model

#
# Function to train rntn
#


def train_rntn(model_name=None, num_samples=None, params=None):
    """Function that trains all models and saves the trained models."""
    data_manager = DataManager()

    clf = RNTN(model_name=model_name)

    if params is None:
        params = {'num_epochs': [10], 'training_rate': [0.0001]}

    score_func = clf.loss()
    if num_samples is None:
        x_train = data_manager.x_train
        y_train = [x_train[i].root.label for i in range(len(x_train))]
        x_dev = data_manager.x_dev
        y_dev = [x_dev[i].root.label for i in range(len(x_dev))]
        x_test = np.asarray(data_manager.x_test).reshape(-1, 1)
        y_test = np.asarray([x_test[i, 0].root.label for i in range(len(x_test))])
    else:
        x_train = data_manager.x_train[0:num_samples]
        y_train = [x_train[i].root.label for i in range(len(x_train))]
        x_dev = data_manager.x_dev[0:int(num_samples*0.2)]
        y_dev = [x_dev[i].root.label for i in range(len(x_dev))]
        x_test = np.asarray(data_manager.x_test[0:int(num_samples*0.2)]).reshape(-1, 1)
        y_test = np.asarray([x_test[i, 0].root.label for i in range(len(x_test))])

    cv = train_model(clf, params, score_func, x_train, y_train, x_dev, y_dev)
    logging.info('Training results: {0}'.format(cv.cv_results_))

    logging.info('Best Model Name: {0}'.format(cv.best_estimator_.model_name))
    logging.info('Best Model Parameters: {0}'.format(cv.best_params_))
    logging.info('Best Model Score: {0}'.format(cv.best_score_))

    y_pred = cv.predict(x_test)
    model_loss = cv.score(x_test.reshape(-1, 1), y_test)
    logging.info("Model Loss (Best Model): {0}".format(model_loss))
    logging.info("Model Accuracy (Best Model): {0}".format(accuracy_score(y_test, y_pred)))

# Call train on run
if __name__ == '__main__':
    train_rntn()
