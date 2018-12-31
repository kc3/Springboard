# -*- coding: utf-8 -*-

#
# Functionality to train all models in this project.
#

from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from src.models.RNTN import RNTN
import random

#
# Singleton to avoid multiple copies.
# Class to hold a copy of data
#


class DataManager:
    """Load interim data for trees as strings."""

    _def_path = './src/data/interim/trainDevTestTrees_PTB/trees/'

    __instance = None

    def __new__(cls, *args, **kwargs):
        """Create the object on first instantiation."""

        if cls.__instance is None:
            cls.__instance = object.__new__(cls)

        return cls.__instance

    def __init__(self, path=None, max_rows=None):
        if path is None:
            path = self._def_path

        if max_rows is not None:
            self._max_rows = max_rows

        # Load data
        self.x_train = self._load(self._make_file_name(path, 'train'), 100)
        self.x_dev = self._load(self._make_file_name(path, 'dev'), 100)
        self.x_test = self._load(self._make_file_name(path, 'test'), 100)

    @staticmethod
    def _make_file_name(file_path, file_name):
        return '{0}{1}.txt'.format(file_path, file_name)

    @staticmethod
    def _load(file_path, max_rows=None):
        """Loads entire content of the file."""

        s = []

        with open(file_path, 'r') as f:
            if max_rows is None:
                return f.read()
            else:
                for i, line in enumerate(f):
                    if i < 100:
                        s.append(line.strip())
                    else:
                        break

            return s


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

    model.fit(x_train+x_dev, y_train+y_dev)

    return model

#
# Function to evaluate a given model
#


def evaluate_model(cv_model, x_test, y_test):
    """Evaluate the model for test set."""
    return cv_model.score(x_test, y_test)

#
# Function to predict labels
#


def predict_model(cv_model, x_test):
    """Predicts labels for given test set."""
    return cv_model.predict(x_test)

#
# Function to train all models
#


def train_all_models():
    """Function that trains all models and saves the trained models."""
    data_manager = DataManager(max_rows=100)

    clf = RNTN()
    params = {}
    score_func = make_scorer(accuracy_score)
    x_train = data_manager.x_train
    y_train = [random.randint(0, 4) for _ in range(len(x_train))]
    x_dev = data_manager.x_dev
    y_dev = [random.randint(0, 4) for _ in range(len(x_dev))]

    cv = train_model(clf, params, score_func, x_train, y_train, x_dev, y_dev)

    y_pred = cv.predict(data_manager.x_test)
    print(y_pred)

    x_test = data_manager.x_test
    y_test = [random.randint(0, 4) for _ in range(len(x_test))]
    score = cv.score(x_test, y_test)
    print(score)
