
import os
import logging
from datetime import datetime
from src.models.data_manager import DataManager


# Function to train SeqToSeq models


def train_seqtoseq(model_name=None, num_samples=None, params=None):
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
    sorted_questions, sorted_answers, questions_int_to_vocab, answers_int_to_vocab = d.get_cornell_data()
    logging.info('Cornell Data Set loaded...')


if __name__ == '__main__':
    train_seqtoseq()
