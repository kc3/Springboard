
import logging
import os
import numpy as np
import tensorflow as tf
from src.models.data_manager import DataManager
from src.models.seqtoseq_model import SeqToSeqModel


class PolicyAgent:
    """Agent using seq2seq model for performing a conversation."""

    def __init__(self,
                 seq2seq_model_name='test-policy',
                 agent_name=None):

        # Model parameters
        self.seq2seq_model_name = seq2seq_model_name
        self.agent_name = agent_name

        # Initialize Data Manager
        self.data_manager = DataManager()

        # Dull responses
        self.dull_responses = self.data_manager.get_cornell_dull_responses()

        logging.info('Agent {0} initialized.'.format(self.agent_name))

    def play(self, state):
        """Play each turn."""
        last_response, request = state

        # Generate new responses
        responses = [
            '{0} {1} {2}'.format(self.agent_name, last_response, 'fizz'),
            '{0} {1} {2}'.format(self.agent_name, last_response, 'buzz')
        ]

        return responses

    def finish(self):
        """Review rewards and optimize network."""
        return -1*np.random.random()

    def save(self):
        """Saves model"""
        pass

    def _loss(self, logits, labels):
        """Loss function used for optimization. This is negative log reward."""
        pass

    def _reward(self):
        """Total reward for this response."""
        pass

    def _ease_of_answering(self, dull_responses, response):
        """Measure for similarity to known dull responses."""
        pass

    def _information_flow(self):
        """Measure for repeating responses."""
        pass

    def _semantic_coherance(self):
        """Measure for conversation flow."""
        pass

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

    def _get_save_dir(self):
        """ Checks for save directory and builds it if necessary.

        :return:
             A string containing save directory path
        """
        abs_path = os.path.abspath(os.path.dirname(__file__))
        def_models_path = os.path.join(abs_path, '../../models/')
        save_dir = def_models_path + self.seq2seq_model_name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return save_dir

    def _get_model_save_path(self):
        """ Builds save path for the model.

        :return:
            A string containing model save path.
        """
        assert self.seq2seq_model_name is not None
        return '{0}/{1}.ckpt'.format(self._get_save_dir(), self.seq2seq_model_name)
