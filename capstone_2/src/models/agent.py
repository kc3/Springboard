
import logging
import tensorflow as tf
from src.models.data_manager import DataManager
from src.models.seqtoseq_model import SeqToSeqModel


class PolicyAgent:
    """Agent using seq2seq model for performing a conversation."""

    def __init__(self,
                 seq2seq_model_name='test-policy',
                 model_name=None):

        # Model parameters
        self.seq2seq_model_name = seq2seq_model_name
        self.model_name = model_name

