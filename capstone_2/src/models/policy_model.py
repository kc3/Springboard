
import logging
import os
import numpy as np
import tensorflow as tf
import time
from src.models.data_manager import DataManager


class PolicyGradientModel:
    """Implementation of the policy gradient model."""

    def __init__(self,
                 turns=5,
                 actions=5,
                 epochs = 100,
                 model_name=None):
        """Model Parameters Init."""

        self.turns = turns
        self.actions = actions
        self.epochs = epochs
        self.model_name = model_name

    def fit(self, data_manager: DataManager):
        """Fits a policy model using a trained seqtoseq model."""

        # Load Agent 1

        # Load Agent 2

        # Create tf Session
        with tf.Graph().as_default(), tf.Session() as session:

            # Initialize Starting prompts

            # For each epoch
            for epoch in range(self.epochs):
                pass

        return

    def predict(self):
        pass

    def _model_inputs(self):
        pass

    def _model_graph(self):
        pass

    def _loss(self, logits, labels):
        pass

    def _reward(self, loss):
        pass

    def _ease_of_answering(self):
        pass

    def _information_flow(self):
        pass

    def _semantic_coherance(self):
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
        save_dir = def_models_path + self.model_name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return save_dir

    def _get_model_save_path(self):
        """ Builds save path for the model.

        :return:
            A string containing model save path.
        """
        assert self.model_name is not None
        return '{0}/{1}.ckpt'.format(self._get_save_dir(), self.model_name)
