
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
                 finish_epochs=10,
                 agent_name=None):

        # Model parameters
        self.seq2seq_model_name = seq2seq_model_name
        self.finish_epochs = finish_epochs
        self.agent_name = agent_name

        # Initialize Data Manager
        self.data_manager = DataManager()

        # Dull responses
        self.dull_responses = self.data_manager.get_cornell_dull_responses()

        # Gather data for finish phase training
        self.questions = []
        self.answers = []
        self.rewards = []

        # Initialize a default graph and session
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        # Create seq2seq model instance
        self.seq2seq_model = SeqToSeqModel(
            epochs=finish_epochs,
            model_name=self.seq2seq_model_name)

        # Create model variables
        cost, train_op, beam_output, encoder_output = self._create_model_graph()
        self.cost = cost
        self.train_op = train_op
        self.beam_output = beam_output
        self.encoder_output = encoder_output

        logging.info('Agent {0} initialized.'.format(self.agent_name))

    def play(self, state):
        """Play each turn."""
        last_response, request = state

        responses = []

        with self.graph.as_default():

            # Get Encoder Output
            encoder_output_prev = self.seq2seq_model.get_encoded_representation(
                self.session, self.encoder_output, last_response, self.data_manager)

            # Predict beam responses
            scores, predicted_ids, parent_ids = self.seq2seq_model.predict_beam_responses(
                self.session, self.beam_output, request, self.data_manager)

            # Prepare responses
            for i in range(self.seq2seq_model.beam_width):
                a_tokens = []
                for j in range(self.seq2seq_model.max_sequence_length):
                    token = predicted_ids[0][j][i]
                    a_tokens.append(token)
                    if token == self.data_manager.answers_vocab_to_int['<EOS>']:
                        break

                responses.append(a_tokens)

            # Add to internal state for finish phase
            for response in responses:
                self.questions.append(request)
                self.answers.append(response)
                self.rewards.append(1.)

        return responses

    def finish(self):
        """Review rewards and optimize graph once conversation is over."""

        # logging.info('Started training agent: {0}'.format(self.agent_name))
        #
        # # Change model name to save agent model state
        # old_model_name = self.seq2seq_model_name
        # self.seq2seq_model.model_name = self.seq2seq_model_name + self.agent_name
        #
        # with self.graph.as_default():
        #     train_loss, valid_loss = self.seq2seq_model.train(
        #         self.session, self.questions, self.answers, self.train_op, self.cost, self.data_manager, save=True)
        #
        # logging.info('Training Loss: {0}, Validation Loss: {1}'.format(train_loss, valid_loss))
        #
        # # Change the name back, best model will overwrite the policy model for next iteration.
        # self.seq2seq_model.model_name = old_model_name

        return -1*np.random.random()

    def save(self):
        """Saves model"""
        # self._save_model(self.session)
        pass

    def close(self):
        """Closes session object"""
        self.session.close()

    def _create_model_graph(self):

        """Creates seq2seq model graph"""
        with self.graph.as_default():

            # Load the model inputs
            input_data, targets, lr, input_sequence_length, output_sequence_length = self.seq2seq_model.model_inputs()

            # Model Graph Variables
            self.seq2seq_model.model_graph_vars(self.data_manager)

            # Create Encoder
            encoder_output, encoder_state = self.seq2seq_model._get_encoder(input_data, input_sequence_length)

            # Create Decoder for Training
            train_logits = self.seq2seq_model._get_decoder_train(
                targets, encoder_output, encoder_state, input_sequence_length, self.data_manager)

            with tf.name_scope("optimization"):
                # Compute weight mask
                mask = tf.sequence_mask(output_sequence_length,
                                        self.seq2seq_model.max_sequence_length, dtype=tf.float32)

                # Loss function
                cost = tf.contrib.seq2seq.sequence_loss(
                    train_logits,
                    targets,
                    mask)

                # Optimizer
                optimizer = tf.train.AdamOptimizer(self.seq2seq_model.learning_rate)

                # Gradient Clipping
                gradients = optimizer.compute_gradients(cost)
                capped_gradients = [
                    (tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
                train_op = optimizer.apply_gradients(capped_gradients)

            # Create beam decoder for generating responses
            beam_output = self.seq2seq_model._get_decoder_infer_beam(
                encoder_output, encoder_state, input_sequence_length, self.data_manager)

            # Initialize the model variables
            self.session.run(tf.global_variables_initializer())

            # Restore session
            saver = tf.train.Saver()
            save_path = self._get_model_save_path()
            saver.restore(self.session, save_path)
            logging.info('Saved model {0} loaded from disk.'.format(save_path))

            logging.info('SeqtoSeq Model initialized for agent {0}'.format(self.agent_name))

            return cost, train_op, beam_output, encoder_output

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
