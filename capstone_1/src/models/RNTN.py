# -*- coding: utf-8 -*-

import logging
import numpy as np
import random
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer
# from sklearn.utils.multiclass import check_classification_targets
# from sklearn.utils.validation import check_X_y, check_is_fitted, check_array

#
# Configure logging
#

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s-%(process)d-%(name)s-%(levelname)s-%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

#
# RNTN.py
# Recursive Tensor Neural Network Implementation.
# Conforms to Estimator interface of scikit-learn.
#


class RNTN(BaseEstimator, ClassifierMixin):
    """Recursive Tensor Neural Network Model. Conforms to Estimator interface of scikit-learn."""

    def __init__(self, embedding_size=10, num_epochs=1):

        #
        # Model Parameters
        #

        # Word embedding size.
        self.embedding_size = embedding_size

        # Number of epochs to run
        self.num_epochs = num_epochs

        #
        # Model State
        # Parameters with trailing underscore are set in fit by convention.
        #

    def _build_model_name(self):
        params = self.get_params()
        params_string = '_'.join(['{0}={1}'.format(arg, value) for arg, value in params.items()])
        return 'RNTN_{0}'.format(params_string)

    def fit(self, x, y=None):
        """Fits model to training samples.

        :param x:
        :param y:
            Labels provided for supervised training. In our case labels are already present in tree.
        :return:
        """

        #
        # Set model name based on parameters
        # This is called here as parameters might be modified outside init using BaseEstimator set_params()
        #
        self.name_ = self._build_model_name()

        #
        # Checks needed for using check_estimator() test.
        # Check that X and y have correct shape
        # x, y = check_X_y(x, y)
        # check_classification_targets(y)

        # Initialize Tensors

        logging.info('Model {0} Training Complete.'.format(self.name_))

        # Return self to conform to interface spec.
        return self

    def predict(self, x):
        """ Predicts class labels for each element in x.

        :param x:
            An array where each element is a tree.
        :return:
            Predicted class label for the tree.
        """

        # Tests for check_estimator()
        # Check is fit had been called
        # check_is_fitted(self, ['name_'])

        # Input validation
        # x = check_array(x)

        return [random.randint(0, 4) for _ in range(len(x))]

    def predict_proba(self, x):
        """ Computes softmax log probabilities for given x.
        Scikit-learn will call this while using self.loss.

        :param x:
            An array where each element is a tree.
        :return:
            Softmax probabilities of each class.
        """
        return [np.exp(-1*random.randint(0, 4)) for _ in x]

    def _loss(self, logits, labels):
        """ Cost function computational graph.

        :param logits:
        :param labels:
        :return:
        """
        return np.sum(np.abs([m - n for m, n in zip(logits, labels)]))

    def loss(self):
        """ Default loss function for this estimator.

        :return:
            A loss function used by GridSearchCV to score the models.
        """
        return make_scorer(self._loss, greater_is_better=False, needs_proba=True)
