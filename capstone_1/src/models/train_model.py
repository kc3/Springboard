# -*- coding: utf-8 -*-

#
# Functionality to train all models in this project.
#

import numpy as np
import random
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

    x = x_train + x_dev
    y = y_train + y_dev
    model.fit(x, y)

    return model

#
# Function to train rntn
#


def train_rntn():
    """Function that trains all models and saves the trained models."""
    data_manager = DataManager()

    clf = RNTN()
    params = {'batch_size': [25], 'num_epochs': [3], 'training_rate': [0.0001]}
    score_func = clf.loss()
    x_train = data_manager.x_train
    y_train = [random.randint(0, 4) for _ in range(len(x_train))]
    x_dev = data_manager.x_dev
    y_dev = [random.randint(0, 4) for _ in range(len(x_dev))]

    cv = train_model(clf, params, score_func, x_train, y_train, x_dev, y_dev)

    y_pred = cv.predict(data_manager.x_test)

    x_test = data_manager.x_test
    y_test = np.array([random.randint(0, 4) for _ in range(len(x_test))]).reshape(-1, 1)
    score = cv.score(x_test, y_test)
    print("Model Accuracy: {0}".format(score))

# Call train on run
if __name__ == '__main__':
    train_rntn()
