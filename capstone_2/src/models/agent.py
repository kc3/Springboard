
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

        # Initialize a default graph and session
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        # Create seq2seq model instance
        self.seq2seq_model = SeqToSeqModel(
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

            # Predict beam responses
            scores, predicted_ids, parent_ids = self.seq2seq_model.predict_beam_responses(
                self.session, self.beam_output, request, self.data_manager)

            # Prepare Probabilities (Softmax)
            exp_scores = np.exp(np.asarray(scores)[-1:][0])
            probs = exp_scores / sum(exp_scores)
            # logging.info('Probabilities: {0}'.format(probs))

            # Prepare responses
            for i in range(self.seq2seq_model.beam_width):
                a_tokens = []
                for j in range(self.seq2seq_model.max_sequence_length):
                    token = predicted_ids[j][i]
                    a_tokens.append(token)
                    if token == self.data_manager.answers_vocab_to_int['<EOS>']:
                        break

                responses.append(a_tokens)

            # Prepare Rewards
            rewards = []
            for idx, response in enumerate(responses):
                rewards.append(self._reward(last_response, request, response, probs[idx]))

        return responses, probs, rewards

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

    def _reward(self, last_response, request, response, prob):
        """Total reward for this response."""
        weights = [0.25, 0.25, 0.5]

        rew_1 = self._ease_of_answering(response, prob)
        rew_2 = self._information_flow(last_response, response)
        rew_3 = self._semantic_coherence(request, response, prob)

        rew_total = np.sum(np.multiply(weights, [rew_1, rew_2, rew_3]))

        # logging.info('Total reward for {0}: {1}'.format(response, rew_total))

        return rew_total

    def _ease_of_answering(self, response, prob):
        """Measure for similarity to known dull responses."""
        t = 0.
        n_a = len(response)

        # Get dull responses
        for dull_resp in self.dull_responses:
            n_s = len(set(dull_resp) & set(response))
            if n_s > 0:
                t += prob / n_s

        t /= n_a
        if t == 0.:
            return 0.
        rew = -np.log(t)

        # logging.info('Ease of answering for {0}: {1}'.format(response, rew))

        return rew

    def _information_flow(self, last_response, response):
        """Measure for repeating responses."""

        if last_response is None:
            return 0.

        # Get Encoder Output for last response
        encoder_output_prev = self.seq2seq_model.get_encoded_representation(
            self.session, self.encoder_output, last_response, self.data_manager)
        prev_fw = encoder_output_prev[0][0]
        prev_bw = encoder_output_prev[1][0]
        prev = np.asarray([prev_fw, prev_bw]).reshape(-1)

        # Get Encoder Output for current response
        encoder_output_curr = self.seq2seq_model.get_encoded_representation(
            self.session, self.encoder_output, response, self.data_manager)
        curr_fw = encoder_output_curr[0][0]
        curr_bw = encoder_output_curr[1][0]
        curr = np.asarray([curr_fw, curr_bw]).reshape(-1)

        # Get Dot Product
        log_prod = -np.log(np.sum(np.multiply(prev, curr)))

        # logging.info('Information flow for {0}: {1}'.format(response, log_prod))

        return log_prod

    @staticmethod
    def _semantic_coherence(request, response, fw_prob):
        """Measure for conversation flow."""
        n_req = len(request)
        n_resp = len(response)

        # Compute backward probability
        # Currently assume this to be the same as fw_prob
        # Need to train a separate NN in backward direction
        rew = np.log(fw_prob) / n_resp + np.log(fw_prob) / n_req

        # logging.info('Semantic Coherence for {0}: {1}'.format(response, rew))

        return rew

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
