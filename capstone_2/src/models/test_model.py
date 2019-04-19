#
# Tests for training and evaluation of all models.
#

import logging
import os
import pytest
import numpy as np
from datetime import datetime
from src.models.data_manager import DataManager
from src.models import train_model
from src.models.predict_model import predict_seqtoseq, predict_seqtoseq_beam
from src.models.agent import PolicyAgent
from src.models.policy_model import PolicyGradientModel

#
# Configure logging
#
abs_path = os.path.abspath(os.path.dirname(__file__))
logs_dir = os.path.join(abs_path, '../../logs')
if not os.path.exists(logs_dir):
    os.mkdir(logs_dir)
    os.chmod(logs_dir, 0o777)
log_path = os.path.join(abs_path, '../../logs/run-{0}.log')
logging.basicConfig(filename=log_path.format(datetime.now().strftime('%Y%m%d-%H%M%S')),
                    level=logging.INFO,
                    format='%(asctime)s-%(process)d-%(name)s-%(levelname)s-%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logging.info('Test Logger Created.')


class TestModel(object):

    def test_get_cornell_data(self):
        d = DataManager()
        x, y, a, b = d.get_cornell_data()
        assert x is not None
        assert y is not None
        assert a is not None
        assert b is not None

    def test_get_starting_prompts(self):
        questions = DataManager().get_cornell_starting_prompts()
        print(questions)
        assert len(questions) > 0

    def test_get_dull_responses(self):
        questions = DataManager().get_cornell_dull_responses()
        print(questions)
        assert len(questions) > 0

    def test_train_seqtoseq(self):
        train_model.train_seqtoseq(model_name='test-seqtoseq')

    def test_predict_seqtoseq(self):
        y = predict_seqtoseq('Hi!', model_name='test-seqtoseq')
        print(y)

    def test_train_policy(self):
        train_model.train_rl(model_name='test-rl')

    def test_predict_policy(self):
        y = predict_seqtoseq_beam('Hi!', model_name='test-seqtoseq-attn')
        print(y)

    def test_policy_reward(self):
        data_manager = DataManager()
        agent = PolicyAgent(agent_name='test-agent')
        last_response = data_manager.question_to_tokens('I am fine. How are you?')
        request = data_manager.question_to_tokens('Great. How is the weather today?')
        responses, probs, rewards = agent.play((last_response, request))
        logging.info(responses)
        logging.info(probs)
        logging.info(rewards)
        assert len(responses) > 0

    @pytest.mark.run_this
    def test_rl_train(self):
        batch_size = 1100
        model = PolicyGradientModel(seq2seq_model_name='test-policy', model_name='rl-policy')
        model_inputs = {
            'request': [[np.random.randint(0, 5000) for _ in range(10)]] * batch_size,
            'response': [[np.random.randint(0, 5000) for _ in range(15)]] * batch_size,
            'probability': [[np.random.random() for _ in range(15)]] * batch_size,
            'reward': [[np.random.random() for _ in range(15)]] * batch_size
        }

        model.train(model_inputs, None)
