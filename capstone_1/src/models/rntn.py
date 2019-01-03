# -*- coding: utf-8 -*-

#
# rntn.py
# Recursive Tensor Neural Network Implementation.
# Conforms to Estimator interface of scikit-learn.
#

import logging
import os
import numpy as np
import random
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer
# from sklearn.utils.multiclass import check_classification_targets
# from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from src.models.data_manager import DataManager
import tensorflow as tf

#
# Configure logging
#

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s-%(process)d-%(name)s-%(levelname)s-%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


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

    def fit(self, x, y=None):
        """Fits model to training samples.
        Called by GridSearchCV to train estimators.

        The model computes word embeddings matrix 'L', of dimensions [d, V], where V is the size of the vocabulary
        and d is the word embedding size.
            Embeddings:
                1. Initialize L using uniform distribution in interval [-r, r] where r = 0.0001.

        A word vector for every intermediate node in the tree, that is not a leaf, is a bi-gram and is computed
        recursively from its children.

        The model implements the following equations for every intermediate node in two steps:
            Composition:
                2. z = W*X + b
                3. a = f(z)

            Projection:
                4. y = U' * a + bs

        where:
            W: Weights to be computed by the model of shape [d, 2*d] for Composition step.
            X: Word embeddings for input words stacked together.  X is a column vector of shape [2*d, 1]
            b: bias term for the node of shape [d, 1]
            f: Non-linear function specifying the compositionality of the classifier. Relu in this case.
            z: Input to non-linear function f.
            a: Output of non-linear function of shape [d, 1]
            U: Weights to be computed by model for Projection Step of shape [d, label_size] where label_size
                is the number of classes.
            bs: Bias term for Projection Step of shape [1, label_size]

        :param x:
            Parsed Trees (training samples)
        :param y:
            Labels provided for supervised training. In our case labels are already present on tree nodes.
        :return:
            self (expected by BaseEstimator interface)
        """

        #
        # Set model name based on parameters
        # This is called here as parameters might be modified outside init using BaseEstimator set_params()
        # Parameters with trailing underscore are set in fit by convention.
        #
        self.name_ = self._build_model_name()

        #
        # Checks needed for using check_estimator() test.
        # Check that X and y have correct shape
        # x, y = check_X_y(x, y)
        # check_classification_targets(y)

        # Initialize a session to run Tensorflow operations on a new graph.
        with tf.Graph().as_default(), tf.Session() as session:

            # Build model graph
            self._build_model_graph()

            # Initialize all variables in this graph
            session.run(tf.global_variables_initializer())

            # Run the optimizer num_epoch times.
            # Each iteration is one full run through the train data set.
            for epoch in range(self.num_epochs):

                # Shuffle data set for every epoch
                train_data_permutation = np.random.permutation(list(range(len(x))))

                # Train using one sample at a time
                for i in train_data_permutation:
                    self._train_tree(x[i])

            # Save model after full run
            saver = tf.train.Saver()
            save_dir = DataManager().def_models_path
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = '{0}/{1}.ckpt'.format(save_dir, self.name_)
            saver.save(session, save_path)

            # Close session
            session.close()

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

    def _build_model_graph(self):
        """ Builds Computational Graph for model state in Tensorflow.

        Defines and initializes the following:
            L: Word embeddings for the vocabulary of shape [d, V]
                where d = word embedding size and V = vocabulary size
            W: Weights to be computed by the model of shape [d, 2*d] for Composition step.
            b: bias term for the node of shape [d, 1]
            U: Weights to be computed by model for Projection Step of shape [d, label_size] where label_size
                is the number of classes.
            bs: Bias term for Projection Step of shape [1, label_size]

        The following variables are built for each training case outside this function.
            f: Non-linear function specifying the compositionality of the classifier. Relu in this case.
            z: Input to non-linear function f.
            a: Output of non-linear function of shape [d, 1]

        :return:
            None.
        """

        uniform_r = 0.0001
        self.label_size_ = 5

        # Get Vocabulary for building word embeddings.
        self._get_vocabulary()

        # Build Word Embeddings.
        with tf.variable_scope('Embeddings'):
            L = tf.get_variable(name='L',
                                shape=[self.embedding_size, self.V_],
                                initializer=tf.random_uniform_initializer(-1*uniform_r, uniform_r))

        # Build Weights and bias term for Composition layer
        with tf.variable_scope('Composition'):
            W = tf.get_variable(name='W',
                                shape=[self.embedding_size, 2*self.embedding_size],
                                initializer=tf.random_uniform_initializer(-1*uniform_r, uniform_r))

            b = tf.get_variable(name='b',
                                shape=[self.embedding_size, 1],
                                initializer=tf.random_uniform_initializer(-1*uniform_r, uniform_r))

        # Build Weights and bias term for Projection Layer
        with tf.variable_scope('Projection'):
            U = tf.get_variable(name='U',
                                shape=[self.embedding_size, self.label_size_],
                                initializer=tf.random_uniform_initializer(-1*uniform_r, uniform_r))

            bs = tf.get_variable(name='bs',
                                 shape=[1, self.label_size_])

    def _train_tree(self, tree):
        """ Trains a single training sample.

        :return:
        """

    def _build_model_name(self):
        """ Builds model name for persistence and retrieval based on model parameters.

        :return:
            String containing model name.
        """

        params = self.get_params()
        params_string = '_'.join(['{0}={1}'.format(arg, value) for arg, value in params.items()])
        return 'RNTN_{0}'.format(params_string)

    def _get_vocabulary(self):
        """ Gets Vocabulary from data_manager.
        Sets internal variables:
            - vocabulary_ (needed for word->index mapping)
            - V_ (vocabulary size)
        :return:
            None.
        """

        # Get DataManager Instance to see the vocabulary
        self.vocabulary_ = DataManager().countvectorizer.vocabulary_
        self.V_ = len(self.vocabulary_)
