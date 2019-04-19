
import logging
import os
import numpy as np
from collections import deque
import tensorflow as tf
from src.models.data_manager import DataManager
from src.models.agent import PolicyAgent
from src.models.seqtoseq_model import SeqToSeqModel


class PolicyGradientModel:
    """Implementation of the policy gradient model."""

    def __init__(self,
                 turns=5,
                 actions=5,
                 epochs=1,
                 seq2seq_model_name='test-policy',
                 model_name=None):
        """Model Parameters Init."""

        self.turns = turns
        self.actions = actions
        self.epochs = epochs
        self.seq2seq_model_name = seq2seq_model_name
        self.model_name = model_name

        # Initialize Data Manager
        self.data_manager = DataManager()

        # Read Starting Prompts
        self.starting_prompts = self.data_manager.get_cornell_starting_prompts()

        # Early stop criteria
        self.early_stop = 1

        logging.info('Policy Gradient Model Initialized.')

    def fit(self):
        """Fits a policy model using a trained seqtoseq model."""

        # Curriculum Learning strategy from 2 moves to max moves
        for turn in range(2, self.turns + 1):

            logging.info('------------------------------------------------------------------------')
            logging.info('Starting training for turn: {0} out of {1}'.format(turn, self.turns))

            # Min game loss
            min_game_loss = None

            # Stop counter.
            stop_cnt = 0

            # Train for num epochs
            for epoch in range(self.epochs):

                logging.info('Starting epoch: {0} out of {1}'.format(epoch+1, self.epochs))

                turn_loss = self.play(turn, min_game_loss)
                logging.info('Turn {0} Epoch {1} loss: {2}'.format(turn, epoch+1, turn_loss))

                if epoch == 0:
                    min_game_loss = turn_loss
                else:
                    if min_game_loss > turn_loss:
                        min_game_loss = turn_loss
                        logging.info('Updated min turn loss: {0}'.format(min_game_loss))
                        stop_cnt = 0
                    else:
                        stop_cnt += 1

                # Early stop if loss has not improved
                if stop_cnt >= self.early_stop:
                    logging.info(
                        'Stopping early at epoch {0} for turn {1}, Min loss: {2}'.format(epoch+1, turn, min_game_loss))
                    break

            logging.info('End training for turn: {0} out of {0}'.format(turn, self.turns))
            logging.info('------------------------------------------------------------------------')

        return

    def play(self, num_turns, min_game_loss, num_agents=2):
        """Plays a game using curriculum learning strategy for the number of turns."""

        # Dictionary to store model inputs for training after a game.
        model_inputs = {
            'request': [],
            'response': [],
            'probability': [],
            'reward': []
        }

        # Create Agents
        agents = {}
        for agent_id in range(num_agents):
            agent_name = 'Agent_{0}'.format(agent_id + 1)
            agents[agent_id] = PolicyAgent(seq2seq_model_name=self.seq2seq_model_name, agent_name=agent_name)

        # Add each starting prompt to initial states
        for prompt in self.starting_prompts:

            logging.info('Starting game...')

            # Queue for tracking moves for each agent
            # Also stores trajectory (state, action, response) tuples
            # Stores (agent_id, last_response, request, turn, trajectory)
            state_queue = deque()

            # Pick a random agent
            agent_id = np.random.randint(0, num_agents)

            # Add starting state
            state_queue.append((agent_id, None, prompt, 0, []))

            # Play
            while len(state_queue) > 0:

                # Pop the first state
                agent_id, last_response, request, turn, trajectory = state_queue.popleft()

                logging.info('Game: Agent: {0}, Turn: {1} : {2}, {3}'.format(agent_id,
                                                                             turn,
                                                                             self._answer_from_tokens(last_response),
                                                                             self._answer_from_tokens(request)))

                # Check if the agent needs to play further
                if turn < num_turns*num_agents:
                    # Get responses from the agent
                    responses, probs, rewards = agents[agent_id].play((last_response, request))

                    # logging.info(
                    #     'Agent returned:\nResponses: {0}\nProbabilities: {1}\nRewards: {2}'.format(
                    #         responses, probs, rewards))

                    # Add states for the next agent
                    next_agent_id = (agent_id + 1) % num_agents
                    for i in range(len(responses)):
                        response = responses[i]
                        trajectory_tuple = (request, response, probs[i], rewards[i])
                        trajectory.append(trajectory_tuple)
                        state_queue.append((next_agent_id, request, response, turn+1, trajectory))
                else:
                    for track in trajectory:
                        request, response, probability, reward = track
                        model_inputs['request'].append(request)
                        model_inputs['response'].append(response)
                        model_inputs['probability'].append(probability)
                        model_inputs['reward'].append(self._reward_to_go(reward))

        # Close agents
        for agent_id in range(num_agents):
            agents[agent_id].close()

        # Train for the next iteration
        game_loss = self.train(model_inputs, min_game_loss)

        return game_loss

    def train(self, model_inputs, min_game_loss):

        # Start the session
        with tf.Graph().as_default() as graph, tf.Session() as session:

            # Create a seq to seq model
            model = SeqToSeqModel(model_name=self.seq2seq_model_name)

            # Load the model inputs
            input_data, targets, lr, input_sequence_length, output_sequence_length = model.model_inputs()
            rewards = graph.get_tensor_by_name('Inputs/rewards:0')

            # Model Graph Variables
            model.model_graph_vars(self.data_manager)

            # Create Encoder
            encoder_output, encoder_state = model._get_encoder(input_data, input_sequence_length)

            # Create Decoder for Training
            train_logits = model._get_decoder_train(
                targets, encoder_output, encoder_state, input_sequence_length, self.data_manager)

            with tf.name_scope("optimization"):
                # Compute weight mask
                mask = tf.sequence_mask(output_sequence_length, model.max_sequence_length, dtype=tf.float32)

                # Loss function
                cost = tf.contrib.seq2seq.sequence_loss(
                    train_logits,
                    targets,
                    mask)

                total_cost = tf.add(cost, tf.reduce_mean(rewards))

                # Optimizer
                optimizer = tf.train.AdamOptimizer(model.learning_rate)

                # Gradient Clipping
                gradients = optimizer.compute_gradients(total_cost)
                capped_gradients = [
                    (tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
                train_op = optimizer.apply_gradients(capped_gradients)

            session.run(tf.global_variables_initializer())

            # Train
            questions = model_inputs['request']
            answers = model_inputs['response']
            rewards = -1*np.multiply(model_inputs['reward'], model_inputs['probability'])
            l_rewards = rewards.tolist()
            for i in range(len(l_rewards)):
                l_rewards[i] = l_rewards[i].tolist()
            summary_train_loss, summary_valid_loss = model.train(
                session, questions, answers, train_op, total_cost, self.data_manager, rewards=l_rewards,
                min_valid_loss=min_game_loss)

            logging.info('Summary Train Loss: \n{0}'.format(summary_train_loss))
            logging.info('Summary Valid Loss: \n{0}'.format(summary_valid_loss))

        return min(summary_valid_loss)

    # Rewards to go
    @staticmethod
    def _reward_to_go(rewards):
        n = len(rewards)
        rtgs = np.zeros_like(rewards)
        for i in reversed(range(n)):
            rtgs[i] = rewards[i] + (rtgs[i + 1] if i + 1 < n else 0)

        rtgs /= len(rewards)
        return rtgs

    def _answer_from_tokens(self, tokens):
        """Converts integers tokens to text."""
        if tokens is None:
            return None

        answer = self.data_manager.answer_from_tokens(tokens)
        return answer

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
        assert self.seq2seq_model_name is not None
        return '{0}/{1}.ckpt'.format(self._get_save_dir(), self.seq2seq_model_name)

    def _build_model(self):
        """Builds the policy model."""
        pass
