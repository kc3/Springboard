
import logging
import os
from datetime import datetime
from src.models.data_manager import DataManager
from src.models.seqtoseq_model import SeqToSeqModel
from src.models.policy_model import PolicyGradientModel


# Function to train SeqToSeq models


def train_seqtoseq(model_name=None, epochs=100):
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

    logging.info('Seq2Seq Model Training started.')

    d = DataManager()
    logging.info('Cornell Data Set loaded...')

    # Train individual agent
    s_model = SeqToSeqModel(model_name=model_name, epochs=epochs)
    s_model.fit(d)
    logging.info('Finished training SeqtoSeq Model...')

    # Train a policy gradient model
    # p_model = PolicyGradientModel()
    # p_model.fit()
    # logging.info('Finished training PolicyGradient Model...')


if __name__ == '__main__':
    train_seqtoseq(model_name='test-seqtoseq-attn')
