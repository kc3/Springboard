# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator, ClassifierMixin
import random

#
# RNTN.py
# Recursive Tensor Neural Network Implementation.
# Conforms to Estimator interface of scikit-learn.
#


class RNTN(BaseEstimator, ClassifierMixin):
    """Recursive Tensor Neural Network Model. Conforms to Estimator interface of scikit-learn."""

    def __init__(self):
        self.name = "RNTN"

    def fit(self, x, y=None):
        # Return self to conform to interface spec.
        return self

    def predict(self, x):
        return [random.randint(0, 4) for _ in range(len(x))]
