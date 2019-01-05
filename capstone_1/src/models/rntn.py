# -*- coding: utf-8 -*-

#
# rntn.py
# Recursive Tensor Neural Network Implementation.
# Conforms to Estimator interface of scikit-learn.
#

from collections import OrderedDict
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

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s-%(process)d-%(name)s-%(levelname)s-%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class RNTN(BaseEstimator, ClassifierMixin):
    """Recursive Tensor Neural Network Model. Conforms to Estimator interface of scikit-learn."""

    def __init__(self, embedding_size=10, num_epochs=1, batch_size=100):

        #
        # Model Parameters
        #

        # Word embedding size.
        self.embedding_size = embedding_size

        # Number of epochs to run
        self.num_epochs = num_epochs

        # Batch size (Number of trees to use in a batch)
        self.batch_size = batch_size

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

        # Get Vocabulary for building word embeddings.
        self._get_vocabulary()

        # Initialize a session to run Tensorflow operations on a new graph.
        with tf.Graph().as_default(), tf.Session() as session:

            # Build placeholders for storing tree node information used to create computational graph.
            self._build_model_placeholders()

            # Build model graph
            self.label_size_ = 5
            self._build_model_graph_var(self.embedding_size, self.V_, self.label_size_)

            # Initialize all variables in this graph
            session.run(tf.global_variables_initializer())

            # Run the optimizer num_epoch times.
            # Each iteration is one full run through the train data set.
            for epoch in range(self.num_epochs):

                # Shuffle data set for every epoch
                np.random.shuffle(x)
                logging.debug('First tree in x:{0}'.format(x[0].text()))

                # Train using batch_size samples at a time
                start_idx = 0
                while start_idx < len(x):
                    end_idx = min(start_idx + self.batch_size, len(x))
                    logging.debug('Processing trees at indices ({0}, {1})'.format(start_idx, end_idx))

                    # Build feed dict
                    feed_dict = self._build_feed_dict(x[start_idx:end_idx])

                    # Build batch graph
                    logits, root_logits = self._build_batch_graph(feed_dict)

                    # Build loss graph
                    loss_graph = self._build_loss_graph(feed_dict)

                    # Build optimizer graph
                    optimizer_graph = self._build_optimizer_graph(loss_graph)

                    # Train
                    # Invoke the graph for optimizer this feed dict.
                    session.run(optimizer_graph, feed_dict=feed_dict)

                    # Loss
                    # Invoke the graph for optimizer this feed dict.
                    session.run(loss_graph, feed_dict=feed_dict)

                    start_idx = end_idx

            # Save model after full run
            #saver = tf.train.Saver()
            #save_path = self._build_save_path()
            #saver.save(session, save_path)

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
        """ Cost function computational graph invocation.

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

    @staticmethod
    def _build_model_graph_var(embedding_size, vocabulary_size, label_size):
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
        :return:
            None.
        """

        uniform_r = 0.0001

        # Build Word Embeddings.
        with tf.variable_scope('Embeddings'):
            _ = tf.get_variable(name='L',
                                shape=[embedding_size, vocabulary_size],
                                initializer=tf.random_uniform_initializer(-1*uniform_r, uniform_r))

        # Build Weights and bias term for Composition layer
        with tf.variable_scope('Composition'):
            _ = tf.get_variable(name='W',
                                shape=[embedding_size, 2*embedding_size],
                                initializer=tf.random_uniform_initializer(-1*uniform_r, uniform_r))

            _ = tf.get_variable(name='b',
                                shape=[embedding_size, 1],
                                initializer=tf.random_uniform_initializer(-1*uniform_r, uniform_r))

            _ = tf.get_variable(name='T',
                                shape=[2*embedding_size, 2*embedding_size, embedding_size],
                                initializer=tf.random_uniform_initializer(-1*uniform_r, uniform_r))

        # Build Weights and bias term for Projection Layer
        with tf.variable_scope('Projection'):
            _ = tf.get_variable(name='U',
                                shape=[embedding_size, label_size],
                                initializer=tf.random_uniform_initializer(-1*uniform_r, uniform_r))

            _ = tf.get_variable(name='bs',
                                shape=[1, label_size])

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

    @staticmethod
    def _build_batch_graph(feed_dict):
        """ Builds Batch graph for this training batch using tf.while_loop from feed_dict.

        This is the main method where both the Composition and Projection Layers are defined
        for each tree node in the feed dict.

        Composition layer is defined using a relu for by default.
        TODO: Add param for other non-linear functions (Example, sigmoid, tanh)

        Projection layer is defined as specified above in fit() documentation. The output of projection
        layer is expected to be probability associated with each sentiment label with a value close to 1
        for expected sentiment label and 0 for others.

        :param feed_dict:
            Feed dictionary to be passed to the batch graph functions

        :return:
            None.
        """

        #
        # Composition functionality
        #

        # Get length of the tensor array
        # squeeze removes dimension of size 1
        n = tf.squeeze(tf.shape(feed_dict['is_leaf']))

        # Function to get word embedding
        def get_word(word_idx):
            with tf.variable_scope('Composition'):
                tf.gather('L', word_idx, axis=1)

        #
        # Composition functions
        #

        def compose_relu(left_child_idx, right_child_idx):
            tensor_left = tf.gather(feed_dict['left_child'], left_child_idx)
            tensor_right = tf.gather(feed_dict['right_child'], right_child_idx)
            X = tf.concat([tensor_left, tensor_right], axis=1)

            # Get model variables
            with tf.variable_scope('Composition'):
                W = tf.gather('W')
                b = tf.gather('b')
                T = tf.gather('T')

            # zs = W * X + b
            zs = tf.add(tf.matmul(W, X), b)

            # zt = X' * T * X
            t_slices = tf.unstack(T, axis=2)
            n_t = tf.shape(T)[2]
            ta_t = tf.TensorArray(tf.float32, size=n_t)

            def cond_t(t_idx, _):
                return tf.less(t_idx, n_t)

            def body_t(t_idx, ta_t):
                return [
                    tf.add(t_idx, 1),
                    ta_t.write(t_idx, tf.matmul(tf.matmul(tf.transpose(X), t_slices[t_idx]), X))
                ]
            zt = tf.stack(tf.while_loop(cond_t, body_t, [0, ta_t]))

            # a = zs + zt
            a = tf.add(zs, zt)

            return tf.nn.relu(a)

        # Define a tensor array to store the final outputs (softmax probabilities)
        tensors = tf.TensorArray(tf.float32,
                                 size=n,
                                 dynamic_size=True,
                                 clear_after_read=False,
                                 infer_shape=False)

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
                                  tf.gather(feed_dict['is_leaf'], i),
                                  # Get Word
                                  get_word(tf.gather(feed_dict['word_index'], i)),
                                  # Else, combine left and right
                                  compose_relu(tensors.read(tf.gather(feed_dict['left_child'], i)),
                                               tensors.read(tf.gather(feed_dict['right_child'], i))),
                              )
                              ),
                tf.add(i, 1)
            ]

        # While loop invocation
        tensors, _ = tf.while_loop(cond, body, [tensors, 0], parallel_iterations=1)

        return tensors

    @staticmethod
    def _build_loss_graph(tensors):
        """ Builds loss function graph.

        Computes the cross entropy loss for sentiment prediction values.

        :param tensors:
            Logit function graph array for every node.

        :return:
            None.
        """

    @staticmethod
    def _build_optimizer_graph(tensors):
        """ Builds optimizer graph.

        This is the primary graph tensor evaluated for training.
        The optimizer is the standard GradientDescentOptimizer in tensorflow.

        :return:
            None.
        """

    def _build_feed_dict(self, trees):
        """ Prepares placeholders with feed dictionary variables.

        1. Flattens the given tree into nodes using post-order traversal.
        2. Adds each node to the placeholders.

        :param trees:
            Trees to process.
        :return:
            OrderedDict containing parameters for every node found in the tree.
        """

        # Values to prepare
        is_leaf_vals = []
        word_index_vals = []
        left_child_vals = []
        right_child_vals = []
        label_vals = []

        start_idx = 0

        # Process all trees
        for tree_idx in range(len(trees)):
            tree_dict = self._tree_feed_data(trees[tree_idx], start_idx)
            is_leaf_vals.append(tree_dict['is_leaf'])
            word_index_vals.append(tree_dict['word_index'])
            left_child_vals.append(tree_dict['left_child'])
            right_child_vals.append(tree_dict['right_child'])
            label_vals.append(tree_dict['label'])
            start_idx += len(tree_dict['is_leaf'])

        # Get Placeholders
        graph = tf.get_default_graph()
        is_leaf = graph.get_tensor_by_name('Inputs/is_leaf:0')
        word_index = graph.get_tensor_by_name('Inputs/word_index:0')
        left_child = graph.get_tensor_by_name('Inputs/left_child:0')
        right_child = graph.get_tensor_by_name('Inputs/right_child:0')
        label = graph.get_tensor_by_name('Inputs/label:0')

        # Create feed dict
        feed_dict = {
            is_leaf: is_leaf_vals,
            word_index: word_index_vals,
            left_child: left_child_vals,
            right_child: right_child_vals,
            label: label_vals
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
        """

        logging.debug('Processing tree: {0}'.format(tree.text()))

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
            nodes_dict[nodes[i]] = i + start_idx

        start_idx += len(nodes)

        # Create feed dict
        feed_dict = {
            'is_leaf': [node.isLeaf for node in nodes],
            'word_index': [self.vocabulary_[node.word] if node.word in self.vocabulary_ else -1 for node in nodes],
            'left_child': [nodes_dict[node.left] if not node.isLeaf else -1 for node in nodes],
            'right_child': [nodes_dict[node.right] if not node.isLeaf else -1 for node in nodes],
            'label': [node.label for node in nodes]
        }

        return feed_dict

    def _build_model_name(self):
        """ Builds model name for persistence and retrieval based on model parameters.

        :return:
            String containing model name.
        """

        params = self.get_params()
        params_string = '_'.join(['{0}'.format(value) for _, value in params.items()])
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

    def _build_save_path(self):
        """ Builds save path for the model

        :return:
            A string containing path.
        """

        save_dir = DataManager().def_models_path + '/' + self.name_
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return '{0}/{1}.ckpt'.format(save_dir, self.name_)
