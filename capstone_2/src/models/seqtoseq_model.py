
from sklearn.base import BaseEstimator
import logging

#
# Sequence to Sequence Model
# LSTM based model with Attention
#


class SeqToSeqModel(BaseEstimator):
    """Implementation of the Sequence to Sequence model."""

    def __init__(self,
                 epochs=100,
                 batch_size=128,
                 rnn_size=512,
                 num_layers=2,
                 encoding_embedding_size=512,
                 decoding_embedding_size=512,
                 learning_rate=0.005,
                 learning_rate_decay=0.9,
                 min_learning_rate=0.0001,
                 keep_probability=0.75,
                 model_name=None
                 ):

        # Model Parameters
        # Number of epochs
        self.epochs = epochs

        # Batch size
        self.batch_size = batch_size

        # Number of units in LSTM cell
        self.rnn_size = rnn_size

        # Number of RNN Layers
        self.num_layers = num_layers

        # Encoder embedding size
        self.encoding_embedding_size = encoding_embedding_size

        # Decoder embedding size
        self.decoding_embedding_size = decoding_embedding_size

        # Learning rate
        self.learning_rate = learning_rate

        # Learning rate decay
        self.learning_rate_decay = learning_rate_decay

        # Minimum Learning rate
        self.min_learning_rate = min_learning_rate

        # Keep probability
        self.keep_probability = keep_probability

        # Model Name
        self.model_name = model_name

        logging.info('Model SeqtoSeq Initialization completed.')

    def fit(self, x, y=None):
        """Training model."""
        pass

    def predict(self, x):
        """Predict function"""
        pass
