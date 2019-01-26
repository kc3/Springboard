# -*- coding: utf-8 -*-

#
# rntn.py
# Recursive Tensor Neural Network Implementation.
# Conforms to Estimator interface of scikit-learn.
#

from collections import OrderedDict
from datetime import datetime
import joblib
import logging
import os
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer, log_loss
from sklearn.preprocessing import OneHotEncoder
# from sklearn.utils.multiclass import check_classification_targets
# from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from src.models.data_manager import DataManager
import tensorflow as tf

#
# Configure logging
#

logging.basicConfig(filename='./logs/run-{0}.log'.format(datetime.now().strftime('%Y%m%d-%H%M%S')),
                    level=logging.INFO,
                    format='%(asctime)s-%(process)d-%(name)s-%(levelname)s-%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class RNTN(BaseEstimator, ClassifierMixin):
    """Recursive Tensor Neural Network Model. Conforms to Estimator interface of scikit-learn."""

    def __init__(self,
                 embedding_size=35,
                 num_epochs=1,
                 batch_size=30,
                 compose_func='tanh',
                 training_rate=0.001,
                 regularization_rate=0.01,
                 label_size=5,
                 model_name=None
                 ):

        #
        # Model Parameters
        #

        # Word embedding size.
        self.embedding_size = embedding_size

        # Number of epochs to run
        self.num_epochs = num_epochs

        # Batch size (Number of trees to use in a batch)
        self.batch_size = batch_size

        # Composition function for neural units
        self.compose_func = compose_func

        # Learning Rate
        self.training_rate = training_rate

        # Regularization Rate
        self.regularization_rate = regularization_rate

        # Label size
        self.label_size = label_size

        # Model Name
        self.model_name = model_name

        logging.info('Model RNTN initialization complete.')

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
                2. zs = W*X + b (Standard term)
                3. zt = X' * T * X (Neural Tensor term)
                4. a = f(zs + zt)

            Projection:
                5. y = U' * a + bs

        where:
            W: Weights to be computed by the model of shape [d, 2*d] for Composition step.
            X: Word embeddings for input words stacked together.  X is a column vector of shape [2*d, 1]
            b: bias term for the node of shape [d, 1]
            T: Tensor of dimension [2*d, 2*d, d]. Each T[:,:,i] slice generates a scalar, which is one component of
                the final word vector of dimension d.
            f: Non-linear function specifying the compositionality of the classifier. Relu in this case.
            zs: Standard RNN term input to non-linear function f of shape [d, 1]
            zt: Neural tensor term input to non-linear function f of shape [d, 1]
            a: Output of non-linear function of shape [d, 1]
            U: Weights to be computed by model for Projection Step of shape [d, label_size] where label_size
                is the number of classes.
            bs: Bias term for Projection Step of shape [label_size, 1]

        :param x:
            Parsed Trees (training samples) in a 2D ndarray of dim (num_samples, 1).
        :param y:
            Labels provided for supervised training.
        :return:
            self (expected by BaseEstimator interface)
        """

        logging.info('Model RNTN fit() called on {0} training samples.'.format(x.shape[0]))
        x = x[:, 0]

        # Set the annealing rate for decreasing learning rate for higher epochs
        annealing_rate = 0.9

        # Early stopping threshold
        early_stop_threshold = 3
        num_bad_epochs = 0

        #
        # Set model name based on parameters
        # This is called here as parameters might be modified outside init using BaseEstimator set_params()
        # Parameters with trailing underscore are set in fit by convention.
        #
        if not self.model_name:
            self.model_name = self._build_model_name(len(x))

        curr_training_rate = self.training_rate
        prev_total_loss = 0.

        #
        # Checks needed for using check_estimator() test.
        # Check that X and y have correct shape
        # x, y = check_X_y(x, y)
        # check_classification_targets(y)

        # Build Vocabulary for word embeddings.
        # This also saves generated vocabulary for predictions.
        self._build_vocabulary(x)

        # Run the optimizer num_epoch times.
        # Each iteration is one full run through the train data set.
        for epoch in range(self.num_epochs):
            logging.info('Epoch {0} out of {1} training started.'.format(epoch, self.num_epochs))

            total_loss = 0.

            # Shuffle data set for every epoch
            np.random.shuffle(x)

            # Train using batch_size samples at a time
            start_idx = 0
            while start_idx < len(x):
                end_idx = min(start_idx + self.batch_size, len(x))
                logging.info('Processing trees at indices ({0}, {1})'.format(start_idx, end_idx))

                # Initialize a session to run tensorflow operations on a new graph.
                with tf.Graph().as_default(), tf.Session() as session:

                    # Create or Load model
                    reset = (epoch == 0 and start_idx == 0)
                    self._load_model(session, reset)

                    # Build feed dict
                    feed_dict = self._build_feed_dict(x[start_idx:end_idx])

                    # Get labels
                    labels = tf.get_default_graph().get_tensor_by_name('Inputs/label:0')

                    # Get length of the tensor array
                    n = tf.squeeze(tf.shape(labels)).eval(feed_dict)
                    logging.info('Feed Dict has {0} labels'.format(n))

                    # Build batch graph
                    logits = self._build_batch_graph(self.get_word, self._get_compose_func())
                    #is_root = tf.get_default_graph().get_tensor_by_name('Inputs/is_root:0')
                    #root_indices = tf.where(is_root)
                    #root_logits = tf.gather(logits, root_indices)
                    #root_labels = tf.gather(labels, root_indices)

                    # Build loss graph
                    loss_tensor = self._build_loss_graph(labels, logits, self.label_size,
                                                         self._regularization_l2_func(self.regularization_rate),
                                                         1.0, feed_dict)

                    # Loss
                    # Invoke the graph for loss function with this feed dict.
                    epoch_loss = loss_tensor.eval(feed_dict)
                    logging.info('Training Loss before balancing = {0}'.format(epoch_loss))

                    # Weights found by manual exploration of all nodes in the graph
                    weights = tf.get_default_graph().get_tensor_by_name('Inputs/weight:0')

                    # Balanced Loss tensor
                    weighted_loss_tensor = self._build_loss_graph(labels, logits, self.label_size,
                                                         self._regularization_l2_func(self.regularization_rate),
                                                         weights, feed_dict)

                    weighted_epoch_loss = weighted_loss_tensor.eval(feed_dict)
                    logging.info('Training Loss after balancing = {0}'.format(weighted_epoch_loss))
                    total_loss += weighted_epoch_loss

                    # Update training loss with the weighted loss
                    train_epoch_loss_val = tf.get_default_graph().get_tensor_by_name('Logging/train_epoch_loss_val:0')
                    if start_idx == 0:
                        # Reset total loss for start of every epoch
                        train_epoch_loss_val = tf.assign(train_epoch_loss_val, weighted_loss_tensor)
                    else:
                        # Update total loss
                        train_epoch_loss_val = tf.assign_add(train_epoch_loss_val, weighted_loss_tensor)

                    logging.debug('Updated total training loss: {0}'.format(train_epoch_loss_val.eval(feed_dict)))

                    # Update training accuracy
                    train_epoch_cum_sum_logits = tf.get_default_graph()\
                        .get_tensor_by_name('Logging/train_epoch_cum_sum_logits:0')
                    train_epoch_accuracy_val = tf.get_default_graph()\
                        .get_tensor_by_name('Logging/train_epoch_accuracy_val:0')

                    y_pred = self._predict_from_logits(logits)
                    labels_int = tf.cast(labels, tf.int32)
                    curr_y_pred_sum = tf.reduce_sum(tf.cast(tf.equal(y_pred, labels_int), tf.float32))
                    accuracy = tf.divide(curr_y_pred_sum, tf.constant(n, dtype=tf.float32))

                    if start_idx == 0:
                        # Reset total accuracy for start of every epoch
                        train_epoch_cum_sum_logits = tf.assign(train_epoch_cum_sum_logits,
                                                               tf.constant(n, dtype=tf.int32))
                        train_epoch_accuracy_val = tf.assign(train_epoch_accuracy_val, accuracy)
                    else:
                        # Update total accuracy
                        past_y_n = train_epoch_cum_sum_logits.eval(feed_dict)
                        past_y_pred_sum = tf.multiply(train_epoch_accuracy_val,
                                                      tf.constant(past_y_n, dtype=tf.float32))
                        total_y_pred_sum = tf.add(past_y_pred_sum, curr_y_pred_sum)
                        train_epoch_cum_sum_logits = tf.assign_add(train_epoch_cum_sum_logits,
                                                                   tf.constant(n, dtype=tf.int32))
                        cumulative_accuracy = tf.divide(total_y_pred_sum,
                                                        tf.constant(past_y_n + n, dtype=tf.float32))
                        train_epoch_accuracy_val = tf.assign(train_epoch_accuracy_val, cumulative_accuracy)

                    logging.debug('Updated total sum logits: {0}'.format(
                        train_epoch_cum_sum_logits.eval(feed_dict)))
                    logging.info('Updated total training accuracy: {0}'.format(
                        train_epoch_accuracy_val.eval(feed_dict)))

                    # Build optimizer graph
                    # Create optimizer
                    all_variables = set(tf.all_variables())
                    optimization_tensor = tf.train.AdagradOptimizer(curr_training_rate).minimize(weighted_loss_tensor)
                    # I honestly don't know how else to initialize adagrad in TensorFlow.
                    session.run(tf.initialize_variables(set(tf.all_variables()) - all_variables))

                    # Train
                    # Invoke the graph for optimizer this feed dict.
                    session.run([optimization_tensor], feed_dict=feed_dict)

                    # Save model after full run
                    # Fit will always overwrite any model
                    self._save_model(session)

                start_idx = end_idx

            logging.info('Total Training Loss: {0} for epoch {1}'.format(total_loss, epoch))

            # Log variables to tensorboard
            self._record_epoch_metrics(epoch)

            # Decrease learning rate if less than previous epoch
            if epoch > 0 and total_loss > prev_total_loss:
                curr_training_rate = curr_training_rate * annealing_rate
                logging.info('Updated Current training rate to {0}'.format(curr_training_rate))
                num_bad_epochs += 1
                if num_bad_epochs >= early_stop_threshold:
                    logging.info('Stopping runs at epoch {0}'.format(epoch))
                    break
            else:
                num_bad_epochs = 0

            prev_total_loss = total_loss

        logging.info('Model {0} Training Complete.'.format(self.model_name))

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
        # check_is_fitted(self, ['vocabulary_'])

        # Input validation
        # x = check_array(x)

        logging.info('Model RNTN predict() called on {0} testing samples.'.format(len(x)))

        y_class_prob = self.predict_proba(x)

        # Get maximum arg val for the class probabilities.
        y_pred = np.argmax(y_class_prob, axis=-1)

        logging.info('Model RNTN predict() completed.')
        return y_pred

    def predict_proba(self, x):
        """ Computes softmax log probabilities for given x.
        Scikit-learn will call this while using self.loss.

        :param x:
            An 2d ndarray where each element is a tree.
        :return:
            Softmax probabilities of each class.
        """

        logging.info('Model RNTN predict_proba() called on {0} testing samples.'.format(x.shape[0]))
        x = x[:, 0]

        # Load vocabulary
        self._load_vocabulary()

        # Initialize a session to run tensorflow operations on a new graph.
        with tf.Graph().as_default(), tf.Session() as session:

            # Load model
            self._load_model(session)

            # Build feed dict
            feed_dict = self._build_feed_dict(x)

            # Build logit functions
            # Get labels
            labels = tf.get_default_graph().get_tensor_by_name('Inputs/label:0')
            is_root = tf.get_default_graph().get_tensor_by_name('Inputs/is_root:0')

            # Get length of the tensor array
            n = tf.squeeze(tf.shape(labels)).eval(feed_dict)
            logging.info('Feed Dict has {0} labels'.format(n))

            # Build batch graph
            logits = self._build_batch_graph(self.get_word, self._get_compose_func())
            root_logits = tf.gather(logits, tf.where(is_root))

            # Get softmax probabilities for the tensors.
            y = tf.squeeze(tf.nn.softmax(root_logits))

            # Evaluate arg vals
            y_prob = y.eval(feed_dict)

        logging.info('Model RNTN predict_proba() returned.')
        return y_prob

    def loss(self):
        """ Default loss function for this estimator.

        :return:
            A loss function used by GridSearchCV to score the models.
        """
        return make_scorer(self._loss, greater_is_better=False, needs_proba=True)

    def _loss(self, actuals, proba):
        """ Cost function called for computing cross entropy loss using numpy.

        :param actuals:
            Actual values (ground truth)
        :param proba:
            Softmax probabilities returned by predict_proba
        :return:
            Computed loss
        """
        encoder = OneHotEncoder(categories=[list(range(self.label_size))], dtype=np.int32)
        actuals_as_x = np.asarray(actuals).reshape(-1, 1)
        labels = encoder.fit_transform(actuals_as_x)
        loss_val = log_loss(actuals, proba, normalize=False, labels=labels)
        logging.info('Model RNTN _loss() returned {0}.'.format(loss_val))

        return loss_val

    @staticmethod
    def _build_model_graph_var(embedding_size, vocabulary_size, label_size, regularization_func):
        """ Builds Computational Graph for model state in Tensorflow.

        Defines and initializes the following:
            L: Word embeddings for the vocabulary of shape [d, V]
                where d = word embedding size and V = vocabulary size
            W: Weights to be computed by the model of shape [d, 2*d] for Composition step.
            T: Tensor of dimension [2*d, 2*d, d]. Each T[:,:,i] slice generates a scalar, which is one component of
                the final word vector of dimension d.
            b: bias term for the node of shape [d, 1]
            U: Weights to be computed by model for Projection Step of shape [d, label_size] where label_size
                is the number of classes.
            bs: Bias term for Projection Step of shape [1, label_size]

        The following variables are built for each training case outside this function.
            f: Non-linear function specifying the compositionality of the classifier. Relu in this case.
            zs: Standard RNN term input to non-linear function f of shape [d, 1]
            zt: Neural tensor term input to non-linear function f of shape [d, 1]
            a: Output of non-linear function of shape [d, 1]

        :param embedding_size:
            Word embedding size
        :param vocabulary_size:
            Vocabulary size
        :param label_size:
            Label size
        :param regularization_func:
            Function used for regularization of weights.
        :return:
            None.
        """

        # Used to initialize word vectors
        uniform_r = 0.0001

        # Build Word Embeddings.
        with tf.variable_scope('Embeddings', reuse=tf.AUTO_REUSE):
            _ = tf.get_variable(name='L',
                                shape=[embedding_size, vocabulary_size],
                                initializer=tf.random_uniform_initializer(-1 * uniform_r, uniform_r),
                                trainable=True)

        # Build Weights and bias term for Composition layer
        with tf.variable_scope('Composition', reuse=tf.AUTO_REUSE):
            _ = tf.get_variable(name='W',
                                shape=[embedding_size, 2 * embedding_size],
                                trainable=True)

            _ = tf.get_variable(name='b',
                                shape=[embedding_size, 1],
                                trainable=True)

            _ = tf.get_variable(name='T',
                                shape=[2 * embedding_size, 2 * embedding_size, embedding_size],
                                trainable=True)

        # Build Weights and bias term for Projection Layer
        with tf.variable_scope('Projection', reuse=tf.AUTO_REUSE):
            _ = tf.get_variable(name='U',
                                shape=[embedding_size, label_size],
                                trainable=True)

            _ = tf.get_variable(name='bs',
                                shape=[label_size, 1],
                                trainable=True)

    @staticmethod
    def _build_model_placeholders():
        """ Builds placeholder nodes used to build computational graph for every tree node.

        :return:
            None.
        """

        with tf.name_scope('Inputs'):
            # Boolean indicating if the node is a leaf
            _ = tf.placeholder(tf.bool, shape=None, name='is_leaf')

            # Int32 indicating word index, -1 for unknown
            _ = tf.placeholder(tf.int32, shape=None, name='word_index')

            # Int32 indicating left child within the flattened tree
            _ = tf.placeholder(tf.int32, shape=None, name='left_child')

            # Int32 indicating right child within the flattened tree
            _ = tf.placeholder(tf.int32, shape=None, name='right_child')

            # Int32 indicating label of the node
            _ = tf.placeholder(tf.int32, shape=None, name='label')

            # Boolean indicating if the node is a root
            _ = tf.placeholder(tf.bool, shape=None, name='is_root')

            # Float32 indicating weight of the node.
            _ = tf.placeholder(tf.float32, shape=None, name='weight')

    @staticmethod
    def _build_model_logging_var():
        """ Builds model logging variables.

        :return:
            None.
        """
        # Build Logging variables.
        with tf.variable_scope('Logging', reuse=tf.AUTO_REUSE):
            train_epoch_loss_val = tf.get_variable(name='train_epoch_loss_val',
                                             shape=(),
                                             trainable=False,
                                             initializer=tf.zeros_initializer)
            train_epoch_accuracy_val = tf.get_variable(name='train_epoch_accuracy_val',
                                                 shape=(),
                                                 trainable=False,
                                                 initializer=tf.zeros_initializer)
            train_epoch_cum_sum_logits = tf.get_variable(name='train_epoch_cum_sum_logits',
                                                 shape=(),
                                                 dtype=tf.int32,
                                                 trainable=False,
                                                 initializer=tf.zeros_initializer)

            dev_epoch_loss_val = tf.get_variable(name='dev_epoch_loss_val',
                                                 shape=(),
                                                 trainable=False,
                                                 initializer=tf.zeros_initializer)
            dev_epoch_accuracy_val = tf.get_variable(name='dev_epoch_accuracy_val',
                                                 shape=(),
                                                 trainable=False,
                                                 initializer=tf.zeros_initializer)

        with tf.name_scope('Logging_Variables'):
            _ = tf.summary.scalar('train_epoch_loss', train_epoch_loss_val)
            _ = tf.summary.scalar('train_epoch_accuracy', train_epoch_accuracy_val)
            _ = tf.summary.scalar('dev_epoch_loss', dev_epoch_loss_val)
            _ = tf.summary.scalar('dev_epoch_accuracy', dev_epoch_accuracy_val)

    # Function to get word embedding
    @staticmethod
    def get_word(word_idx):
        """ Get word embedding from model variable.

        :param word_idx:
            Index of the word from vocabulary.
        :return:
            The word embedding.
        """
        with tf.variable_scope('Embeddings', reuse=True):
            L = tf.get_variable('L')

        word = tf.cond(tf.less(word_idx, 0),
                       lambda: tf.random_uniform(tf.gather(L, 0, axis=1).shape, -0.0001, maxval=0.0001),
                       lambda: tf.gather(L, word_idx, axis=1))
        word_col = tf.expand_dims(word, axis=1)
        return word_col

    # Function to build composition function for a single non leaf node
    @staticmethod
    def compose_func_helper(X):
        """ Composes graph for intermediate nodes.

        :param X:
            Concatenated vector for both children.
        :return:
            Composition Layer Input to be used in compose_func.
        """
        # Get model variables
        with tf.variable_scope('Composition', reuse=True):
            W = tf.get_variable('W')
            b = tf.get_variable('b')
            T = tf.get_variable('T')

        # zs = W * X + b
        zs = tf.add(tf.matmul(W, X), b)

        # zt = X' * T * X
        m1 = tf.tensordot(T, X, [[1], [0]])
        m2 = tf.tensordot(X, m1, [[0], [0]])
        zt = tf.expand_dims(tf.squeeze(m2), axis=1)

        # a = zs + zt
        a = tf.add(zs, zt)
        return a

    def compose_relu(self, X):
        """ Rectified Linear Unit Composition.

        :param X:
            Concatenated vector of both children.
        :return:
            Logit after composition.
        """
        return tf.nn.relu(self.compose_func_helper(X))

    def compose_tanh(self, X):
        """ Tanh Composition.

        :param X:
            Concatenated vector of both children.
        :return:
            Logit after composition.
        """
        return tf.nn.tanh(self.compose_func_helper(X))

    def _get_compose_func(self):
        """ Gets composition function by name from model parameter compose_func.
        :return:
            Composition function.
        """
        if self.compose_func == 'relu':
            compose_func_p = self.compose_relu
        else:
            if self.compose_func == 'tanh':
                compose_func_p = self.compose_tanh
            else:
                raise ValueError("Unknown Composition Function: {0}".format(self.compose_func))

        return compose_func_p

    @staticmethod
    def _build_batch_graph(get_word_func, compose_func):
        """ Builds Batch graph for this training batch using tf.while_loop from feed_dict.

        This is the main method where both the Composition and Projection Layers are defined
        for each tree node in the feed dict.

        Composition layer is defined using a relu for by default.

        Projection layer is defined as specified above in fit() documentation. The output of projection
        layer is expected to be probability associated with each sentiment label with a value close to 1
        for expected sentiment label and 0 for others.

        :param get_word_func:
            Function that will be evaluated to get word embedding.
        :param compose_func:
            Function that will be evaluated to compose two vectors.
        :return logits:
            An array of tensors containing unscaled probabilities for all nodes.
        """

        # Get Placeholders
        graph = tf.get_default_graph()
        is_leaf = graph.get_tensor_by_name('Inputs/is_leaf:0')
        word_index = graph.get_tensor_by_name('Inputs/word_index:0')
        left_child = graph.get_tensor_by_name('Inputs/left_child:0')
        right_child = graph.get_tensor_by_name('Inputs/right_child:0')

        # Get length of the tensor array
        # squeeze removes dimension of size 1
        n = tf.squeeze(tf.shape(is_leaf))

        # Define a tensor array to store the logits (outputs from projection layer)
        tensors = tf.TensorArray(tf.float32,
                                 size=n,
                                 clear_after_read=False)

        # Define loop condition
        # node_idx < len(tensors)
        cond = lambda tensors, node_idx: tf.less(node_idx, n)

        # Define loop body
        # Defines the
        # If leaf return word embedding else combine left and right tensors
        body = lambda tensors, i:\
            [
                tensors.write(i,
                              tf.cond(
                                  # If Leaf
                                  tf.gather(is_leaf, i),
                                  # Get Word
                                  lambda: get_word_func(tf.gather(word_index, i)),
                                  # Else, combine left and right
                                  lambda: compose_func(tf.concat(
                                      [
                                          tensors.read(tf.gather(left_child, i)),
                                          tensors.read(tf.gather(right_child, i))
                                      ], axis=0)))),
                tf.add(i, 1)
            ]

        # While loop invocation
        tensors, _ = tf.while_loop(cond, body, [tensors, 0], parallel_iterations=1)

        # Concatenate and reshape tensor array for projection
        p = tf.transpose(tf.reshape(tf.squeeze(tensors.concat()), [n, -1]))

        # Add projection layer
        with tf.variable_scope('Projection', reuse=True):
            U = tf.get_variable('U')
            bs = tf.get_variable('bs')

        logits = tf.transpose(tf.matmul(tf.transpose(U), p) + bs)
        return logits

    @staticmethod
    def _regularization_l2_func(regularization_rate):
        """ Regularization function.

        :param regularization_rate:
            Regularization rate.
        :return:
            Lambda function (tensor) -> tensor.
        """
        return lambda x: tf.multiply(tf.nn.l2_loss(x), regularization_rate)

    @staticmethod
    def _build_loss_graph(labels, logits, label_size, regularization_func, weights, feed_dict):
        """ Builds loss function graph.

        Computes the cross entropy loss for sentiment prediction values.

        :param labels:
            Ground truth labels.
        :param logits:
            Logits (unscaled probabilities) for every node.
        :param label_size:
            Size of each label.
        :param regularization_func:
            Regularization function for weights.
        :param weights:
            Weight for balancing loss.
        :return:
            Loss tensor for the whole network.
        """

        # Exclude labels with value 2 while computing loss.
        # This is needed to get around class imbalance problem.
        #idx = tf.where(tf.less(labels, 2))
        #labels_chosen = tf.gather(labels, idx)
        #logits_chosen = tf.gather(logits, idx)

        # stop_gradient stops backprop for labels
        labels_encoded = tf.one_hot(labels, label_size)
        labels_no_grad = tf.stop_gradient(labels_encoded)

        # Get Cross Entropy Loss
        cross_entropy_loss = tf.losses.softmax_cross_entropy(labels_no_grad, logits, weights=weights)
        #cross_entropy_loss = tf.reduce_sum(cross_entropy)
        logging.info('Cross Entropy Loss: {0}'.format(cross_entropy_loss.eval(feed_dict)))

        # Get Regularization Loss for weight terms excluding biases
        with tf.variable_scope('Embeddings', reuse=True):
            L = tf.get_variable('L')

        with tf.variable_scope('Composition', reuse=True):
            W = tf.get_variable('W')
            T = tf.get_variable('T')

        with tf.variable_scope('Projection', reuse=True):
            U = tf.get_variable('U')

        regularization_embedding_loss = regularization_func(L)
        regularization_composition_loss = tf.add(regularization_func(W), regularization_func(T))
        regularization_projection_loss = regularization_func(U)
        regularization_loss = tf.add(regularization_embedding_loss,
                                     tf.add(regularization_composition_loss, regularization_projection_loss))
        logging.info('Regularization Loss: {0}'.format(regularization_loss.eval(feed_dict)))

        # Return Total Loss
        total_loss = tf.add(cross_entropy_loss, regularization_loss)
        return total_loss

    def _build_feed_dict(self, trees):
        """ Prepares placeholders with feed dictionary variables.

        1. Flattens the given tree into nodes using post-order traversal.
        2. Adds each node to the placeholders.

        :param trees:
            Trees to process.
        :return feed_dict:
            OrderedDict containing parameters for every node found in the tree.
        """

        # Values to prepare
        is_leaf_vals = []
        word_index_vals = []
        left_child_vals = []
        right_child_vals = []
        label_vals = []
        is_root_vals = []
        weight_vals = []

        start_idx = 0

        # Process all trees
        for tree_idx in range(len(trees)):
            tree_dict = self._tree_feed_data(trees[tree_idx], start_idx)
            is_leaf_vals.extend(tree_dict['is_leaf'])
            word_index_vals.extend(tree_dict['word_index'])
            left_child_vals.extend(tree_dict['left_child'])
            right_child_vals.extend(tree_dict['right_child'])
            label_vals.extend(tree_dict['label'])
            is_root_vals.extend(tree_dict['is_root'])
            weight_vals.extend(tree_dict['weight'])
            start_idx += len(tree_dict['is_leaf'])

        # Check whether tensors are written before read.
        assert np.all([left_child_vals[i] < i for i in range(len(left_child_vals))])
        assert np.all([right_child_vals[i] < i for i in range(len(right_child_vals))])

        # Get Placeholders
        graph = tf.get_default_graph()
        is_leaf = graph.get_tensor_by_name('Inputs/is_leaf:0')
        word_index = graph.get_tensor_by_name('Inputs/word_index:0')
        left_child = graph.get_tensor_by_name('Inputs/left_child:0')
        right_child = graph.get_tensor_by_name('Inputs/right_child:0')
        label = graph.get_tensor_by_name('Inputs/label:0')
        is_root = graph.get_tensor_by_name('Inputs/is_root:0')
        weight = graph.get_tensor_by_name('Inputs/weight:0')

        # Create feed dict
        feed_dict = {
            is_leaf: is_leaf_vals,
            word_index: word_index_vals,
            left_child: left_child_vals,
            right_child: right_child_vals,
            label: label_vals,
            is_root: is_root_vals,
            weight: weight_vals
        }

        return feed_dict

    def _tree_feed_data(self, tree, start_idx):
        """ Gets feed data for a single tree.

        :param tree:
            Tree to get data for.
        :param start_idx:
            Start index of the node id.
        :return:
            Returns a dict containing the following data:
                1. is_leaf - Boolean array indicating whether the node is leaf or intermediate node.
                2. word_index - Array indicating vocabulary index of the word for leaf nodes or -1 otherwise.
                3. left_child - Array of left children indices or -1 for leaf nodes.
                4. right_child - Array of right children indices or -1 for leaf nodes.
                5. label = Array of labels indicating sentiment label.
                6. is_root = Boolean array indicating whether the node is a root node.
                7. weight = Weight used for balancing training data.
        """

        logging.debug('Processing tree: {0}'.format(tree.text()))

        weights = [124.26106195, 18.01347017, 1., 12.18457133, 52.02482401]

        # Flatten tree into a list using a stack
        nodes_dict = OrderedDict()
        nodes = []
        stack = [tree.root]

        while stack:
            node = stack.pop()
            if not node.isLeaf:
                stack.append(node.left)
                stack.append(node.right)
            # Insert at zero or if using append reverse to ensure children come before parent.
            nodes.insert(0, node)

        for i in range(len(nodes)):
            nodes_dict[nodes[i]] = i

        # Create feed dict
        feed_dict = {
            'is_leaf': [node.isLeaf for node in nodes],
            'word_index': [self._get_word_index(node.word) for node in nodes],
            'left_child': [nodes_dict[node.left]+start_idx if not node.isLeaf else -1 for node in nodes],
            'right_child': [nodes_dict[node.right]+start_idx if not node.isLeaf else -1 for node in nodes],
            'label': [node.label for node in nodes],
            'is_root': [True if i == len(nodes)-1 else False for i in range(len(nodes))],
            'weight': [weights[node.label] for node in nodes]
        }

        # checks
        for i in range(len(nodes)):
            node = nodes[i]
            assert feed_dict['is_leaf'][i] == node.isLeaf
            if node.isLeaf:
                assert feed_dict['left_child'][i] == -1
                assert feed_dict['right_child'][i] == -1
            else:
                assert feed_dict['word_index'][i] == -1
                assert start_idx <= feed_dict['left_child'][i] < start_idx + i, \
                    'Left:{0}'.format(feed_dict['left_child'])
                assert start_idx <= feed_dict['right_child'][i] < start_idx + i, \
                    'Right:{0}'.format(feed_dict['right_child'])
            assert 0 <= feed_dict['label'][i] <= 4
            if i == len(nodes) - 1:
                assert feed_dict['is_root'][i]
            else:
                assert not feed_dict['is_root'][i]

        return feed_dict

    def _build_model_name(self, num_samples):
        """ Builds model name for persistence and retrieval based on model parameters.

        :param num_samples:
            Number of samples used for training.
        :return:
            String containing model name.
        """

        params = self.get_params()
        params_string = '_'.join(['{0}'.format(value) for _, value in params.items()])
        return 'RNTN_{0}_{1}'.format(params_string, num_samples)

    def _get_save_dir(self):
        """ Checks for save directory and builds it if necessary.

        :return:
             A string containing save directory path
        """
        save_dir = DataManager().def_models_path + '/' + self.model_name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return save_dir

    def _get_model_save_path(self):
        """ Builds save path for the model.

        :return:
            A string containing model save path.
        """
        return '{0}/{1}.ckpt'.format(self._get_save_dir(), self.model_name)

    def _get_vocabulary_save_path(self):
        """ Builds save path for the vocabulary for the model.

        :return:
            A string containing model save path.
        """
        return '{0}/vocabulary.pkl'.format(self._get_save_dir(), self.model_name)

    def _build_vocabulary(self, trees):
        """ Builds a dictionary of vocabulary words and persists it.

        :param trees:
            Collection of trees.
        :return:
            None.
        """
        # Build a dictionary of words
        self.vocabulary_ = {}
        word_index = 0

        # Parse trees and add to dict.
        for i in range(len(trees)):
            tree = trees[i]

            # Add tree leaves recursively to the vocabulary
            stack = [tree.root]
            while stack:
                node = stack.pop()
                if node.isLeaf:
                    if node.word not in self.vocabulary_:
                        self.vocabulary_[node.word] = word_index
                        word_index = len(self.vocabulary_)
                else:
                    stack.append(node.right)
                    stack.append(node.left)

        logging.info('Built dictionary for model {0} of size {1}'.format(self.model_name, len(self.vocabulary_)))

        # Save
        save_path = self._get_vocabulary_save_path()
        joblib.dump(self.vocabulary_, save_path, compress=9)
        logging.info('Saved dictionary to {0}'.format(self.model_name))

        self.V_ = len(self.vocabulary_)

    def _load_vocabulary(self):
        """ Loads the vocabulary from the disk

        :return:
            None.
        """
        save_path = self._get_vocabulary_save_path()

        # Load the dictionary
        if not os.path.exists(save_path):
            raise IOError('Vocabulary not found ast {0}. Please train the model using fit() first.'.format(save_path))

        self.vocabulary_ = joblib.load(save_path)
        self.V_ = len(self.vocabulary_)
        logging.info('Loaded dictionary from {0} of size {1}'.format(save_path, self.V_))

    def _get_word_index(self, word):
        """ Returns the word index for a given word.

        :param word:
            Word to search in existing vocabulary for.
        :return:
            Word index if it is present in vocabulary or -1 otherwise.
        """
        # Load vocabulary if necessary
        if not hasattr(self, 'vocabulary_'):
            self._load_vocabulary()

        assert hasattr(self, 'vocabulary_')

        if word is not None and word in self.vocabulary_:
            return self.vocabulary_[word]

        return -1

    def _load_model(self, session, reset=False):
        """ Loads model from disk into session variables

        :param session:
            valid session object.
        :return:
            None
        """

        # Create placeholders
        self._build_model_placeholders()

        # Build model graph
        self._build_model_graph_var(self.embedding_size, self.V_, self.label_size,
                                    self._regularization_l2_func(self.regularization_rate))

        # Build logging variables
        self._build_model_logging_var()

        # Initialize all variables in this graph
        session.run(tf.global_variables_initializer())

        # Load model
        if not reset:
            saver = tf.train.Saver()
            save_path = self._get_model_save_path()
            saver.restore(session, save_path)
            logging.info('Saved model {0} loaded from disk.'.format(save_path))

    def _save_model(self, session):
        """ Saves model to the disk. Should be called only by fit.

        :param session:
            Valid session object.
        :return:
            None.
        """

        # Save model for tensorflow reuse for next epoch
        saver = tf.train.Saver()
        save_path = self._get_model_save_path()
        saver.save(session, save_path)

    def predict_proba_full_tree(self, x):
        """ Computes the prediction for each node in the tree.

        :param x:
            An 2d ndarray where each element is a tree.
        :return y_prob:
            Softmax probabilities of each class for each tree node.
        """

        logging.info('Model RNTN predict_full_tree() called on {0} testing samples.'.format(x.shape[0]))
        x = x[:, 0]

        # Load vocabulary
        self._load_vocabulary()

        # Initialize a session to run tensorflow operations on a new graph.
        with tf.Graph().as_default(), tf.Session() as session:

            # Load model
            self._load_model(session)

            # Build feed dict
            feed_dict = self._build_feed_dict(x)

            # Build logit functions
            # Get labels
            labels = tf.get_default_graph().get_tensor_by_name('Inputs/label:0')

            # Get length of the tensor array
            n = tf.squeeze(tf.shape(labels)).eval(feed_dict)
            logging.info('Feed Dict has {0} labels'.format(n))

            # Build batch graph
            logits = self._build_batch_graph(self.get_word, self._get_compose_func())

            # Get softmax probabilities for the tensors.
            y = tf.squeeze(tf.nn.softmax(logits))

            # Evaluate values
            y_prob = y.eval(feed_dict)

        logging.info('Model RNTN predict_full_tree() returned.')
        return y_prob

    def _predict_from_logits(self, logits):
        """ Returns a tensor that makes predictions from logits.

        :param logits:
            Unscaled probabilities output from the neural network.
        :return:
            Tensor for making predictions.
        """
        # Get softmax probabilities for the tensors.
        y_class_prob = tf.squeeze(tf.nn.softmax(logits))

        # Get maximum arg val for the class probabilities.
        y_pred = tf.argmax(y_class_prob, axis=-1, output_type=tf.int32)

        return y_pred

    def _record_epoch_metrics(self, epoch):
        """ Evaluate current epoch metrics.

        :param epoch:
            Epoch num of the training run.
        :return:
            None.
        """
        x_dev = DataManager().x_dev
        y_dev = [x_dev[i].root.label for i in range(len(x_dev))]

        logging.info('Model RNTN _record_epoch_metrics() called on {0} testing samples.'.format(len(x_dev)))

        # Initialize a session to run tensorflow operations on a new graph.
        with tf.Graph().as_default(), tf.Session() as session:

            # Load model
            self._load_model(session)

            # Build feed dict
            feed_dict = self._build_feed_dict(x_dev)

            # Build logit functions
            # Get labels
            labels = tf.get_default_graph().get_tensor_by_name('Inputs/label:0')

            # Get length of the tensor array
            n = tf.squeeze(tf.shape(labels)).eval(feed_dict)
            logging.info('Feed Dict has {0} labels'.format(n))

            # Build batch graph
            logits = self._build_batch_graph(self.get_word, self._get_compose_func())
            #is_root = tf.get_default_graph().get_tensor_by_name('Inputs/is_root:0')
            #root_indices = tf.where(is_root)
            #root_logits = tf.gather(logits, root_indices)
            #root_labels = tf.gather(labels, root_indices)

            # Build loss graph
            loss_tensor = self._build_loss_graph(labels, logits, self.label_size,
                                                 self._regularization_l2_func(self.regularization_rate),
                                                 1.0, feed_dict)

            # Update loss
            dev_epoch_loss_val = tf.get_default_graph().get_tensor_by_name('Logging/dev_epoch_loss_val:0')
            dev_epoch_loss_val = tf.assign(dev_epoch_loss_val, loss_tensor)
            logging.info('Cross Validation Loss after optimization = {0}'.format(dev_epoch_loss_val.eval(feed_dict)))

            # Get predictions for the tensors.
            y_pred = self._predict_from_logits(logits)
            labels_int = tf.cast(labels, tf.int32)
            curr_y_pred_sum = tf.reduce_sum(tf.cast(tf.equal(y_pred, labels_int), tf.float32))
            accuracy = tf.divide(curr_y_pred_sum, tf.constant(n, dtype=tf.float32))

            dev_epoch_accuracy_val = tf.get_default_graph().get_tensor_by_name('Logging/dev_epoch_accuracy_val:0')
            dev_epoch_accuracy_val = tf.assign(dev_epoch_accuracy_val, accuracy)
            logging.info('Cross Validation Accuracy after optimization = {0}'.format(
                dev_epoch_accuracy_val.eval(feed_dict)))

            # Record Summary operation
            merge = tf.summary.merge_all()

            # Create log file writer to record training progress.
            # Logs can be viewed by running in cmd window: "tensorboard --logdir logs"
            training_writer = tf.summary.FileWriter("./logs/{}/training".format(self.model_name), session.graph)

            summary = session.run(merge)

            # Write the current training status to the log files
            training_writer.add_summary(summary, epoch)
            logging.info('Model RNTN _record_epoch_metrics() returned.')

