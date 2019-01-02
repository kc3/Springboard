# -*- coding: utf-8 -*-

import numpy as np
import random
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer

#
# RNTN.py
# Recursive Tensor Neural Network Implementation.
# Conforms to Estimator interface of scikit-learn.
#


class RNTN(BaseEstimator, ClassifierMixin):
    """Recursive Tensor Neural Network Model. Conforms to Estimator interface of scikit-learn."""

    def __init__(self):

        # Model Name
        self.name = "RNTN"

        # Model Parameters

        #
        # Model State
        #

        # Loss function
        self.loss = make_scorer(self._loss, greater_is_better=False, needs_proba=True)

    def fit(self, x, y=None):
        """Fits model to training samples.

        :param x:
        :param y:
        :return:
        """

        # Initialize Tensors

        # Return self to conform to interface spec.
        return self

    def predict(self, x):
        """ Predicts class labels for each element in x.

        :param x:
            An array where each element is a tree.
        :return:
            Predicted class label for the tree.
        """
        return [random.randint(0, 4) for _ in range(len(x))]

    def predict_proba(self, x):
        """ Computes softmax log probabilities for given x.

        :param x:
            An array where each element is a tree.
        :return:
            Softmax probabilities of each class.
        """
        print("X size: {0}".format(len(x)))
        print("First Tree text: {0}".format(x[0].text()))
        return [t.root.label for t in x]

    def _loss(self, logits, labels):
        """ Cost function computational graph.

        :param logits:
        :param labels:
        :return:
        """
        return np.sum(np.abs([m - n for m, n in zip(logits, labels)]))
