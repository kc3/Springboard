
import logging
import numpy as np
from collections import deque
from src.models.data_manager import DataManager
from src.models.agent import PolicyAgent


class PolicyGradientModel:
    """Implementation of the policy gradient model."""

    def __init__(self,
                 turns=2,
                 actions=5,
                 epochs=100,
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

            # Min turn loss
            min_turn_loss = 0.

            # Stop counter.
            stop_cnt = 0

            # Train for num epochs
            for epoch in range(self.epochs):

                logging.info('Starting epoch: {0} out of {1}'.format(epoch+1, self.epochs))

                turn_loss = self.play(turn, min_turn_loss)
                logging.info('Turn {0} Epoch {1} loss: {2}'.format(turn, epoch+1, turn_loss))

                if epoch == 0:
                    min_turn_loss = turn_loss
                else:
                    if min_turn_loss > turn_loss:
                        min_turn_loss = turn_loss
                        logging.info('Updated min turn loss: {0}'.format(min_turn_loss))
                        stop_cnt = 0
                    else:
                        stop_cnt += 1

                # Early stop if loss has not improved
                if stop_cnt >= self.early_stop:
                    logging.info(
                        'Stopping early at epoch {0} for turn {1}, Min loss: {2}'.format(epoch+1, turn, min_turn_loss))
                    break

            logging.info('End training for turn: {0} out of {0}'.format(turn, self.turns))
            logging.info('------------------------------------------------------------------------')

        return

    def play(self, num_turns, min_turn_loss, num_agents=2):
        """Plays a game using curriculum learning strategy for the number of turns."""

        # Games losses
        game_losses = {}

        # Create Agents
        agents = {}
        for agent_id in range(num_agents):
            agent_name = 'Agent_{0}'.format(agent_id + 1)
            agents[agent_id] = PolicyAgent(seq2seq_model_name=self.seq2seq_model_name, agent_name=agent_name)
            game_losses[agent_id] = 0.

        # Add each starting prompt to initial states
        for prompt in self.starting_prompts:

            logging.info('Starting game...')

            # Queue for tracking moves for each agent
            # Stores (agent_id, last_response, request, turn)
            state_queue = deque()

            # Pick a random agent
            agent_id = np.random.randint(0, num_agents)

            # Add starting state
            state_queue.append((agent_id, None, prompt, 0))

            # Play
            while len(state_queue) > 0:

                # Pop the first state
                agent_id, last_response, request, turn = state_queue.popleft()

                logging.info('Game: Agent: {0}, Turn: {1} : {2}, {3}'.format(agent_id, turn, last_response, request))

                # Check if the agent needs to play further
                if turn < num_turns*num_agents:
                    # Get responses from the agent
                    responses = agents[agent_id].play((last_response, request))

                    # Add states for the next agent
                    next_agent_id = (agent_id + 1) % num_agents
                    for response in responses:
                        state_queue.append((next_agent_id, request, response, turn+1))

            # Finish
            for agent_id in range(num_agents):
                game_losses[agent_id] += agents[agent_id].finish()

            logging.info('Game losses by agent: {0}'.format(game_losses))

        # Save model if best turn yet
        total_losses = sum(game_losses.values())
        logging.info('Total losses: {0}, Previous Min loss: {1}'.format(total_losses, min_turn_loss))
        if total_losses < min_turn_loss:
            best_agent = 0
            best_score = game_losses[0]
            for agent_id in range(num_agents):
                if game_losses[agent_id] < best_score:
                    best_agent = agent_id
                    best_score = game_losses[agent_id]
            agents[best_agent].save()
            logging.info('Saved model for best agent: {0} with loss: {1}'.format(best_agent, best_score))

        return total_losses
