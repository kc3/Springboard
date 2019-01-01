# -*- coding: utf-8 -*-

#
# Functionality to train all models in this project.
#

import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from src.features.tree import Tree
from src.models.RNTN import RNTN


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

        # Load data and store them as trees
        self.x_train = self._load(self._make_file_name(path, 'train'), max_rows)
        self.x_dev = self._load(self._make_file_name(path, 'dev'), max_rows)
        self.x_test = self._load(self._make_file_name(path, 'test'), max_rows)

        # Build Corpus
        self.countvectorizer = CountVectorizer()
        self._build_corpus()

    @staticmethod
    def _make_file_name(file_path, file_name):
        return '{0}{1}.txt'.format(file_path, file_name)

    @staticmethod
    def _load(file_path, max_rows=None):
        """Loads entire content of the file."""
        s = []
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if max_rows is None or i < max_rows:
                    tree_string = line.strip()
                    t = Tree(tree_string)
                    s.append(t)
                else:
                    break

            return s

    def _build_corpus(self):
        """Builds corpus from tree strings"""
        x = self.x_train + self.x_dev
        corpus = []
        for i in range(len(x)):
            corpus.append(x[i].text())

        # Use CountVectorizer to build dictionary of words.
        self.countvectorizer.fit(corpus)

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
# Function to train all models
#


def train_all_models():
    """Function that trains all models and saves the trained models."""
    data_manager = DataManager()

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
