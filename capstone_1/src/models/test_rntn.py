# -*- coding: utf-8 -*-

#
# Tests for training and evaluation of RNTN models.
#

# from sklearn.utils.estimator_checks import check_estimator
#from src.models import train_model
from src.models.rntn import RNTN
from src.models.data_manager import DataManager
import tensorflow as tf


class TestRNTN(object):

    #def __init__(self):
    #    self.data_mgr = DataManager()

    # This test is failing type checks due to tree data structure. Enable once fixed.
    # def test_rntn_estimator(self):
    #    check_estimator(train_model.RNTN)

    #def test_construction(self):
    #    r = RNTN()
    #    assert r is not None

    def test_fit(self):
        data_mgr = DataManager()
        with tf.Session() as s:
            r = RNTN()
            x = data_mgr.x_train
            r.fit(x, None)

    # def test_word(self):
    #     data_mgr = DataManager()
    #
    #     with tf.Session() as s:
    #         r = RNTN()
    #         r._get_vocabulary()
    #         r.label_size_ = 5
    #         r._build_model_graph_var(r.embedding_size, r.V_, r.label_size_)
    #         s.run(tf.global_variables_initializer())
    #         t = r.get_word(23)
    #         assert t is not None
    #         assert t.shape == [r.embedding_size, 1]

    # def test_word_missing(self):
    #     data_mgr = DataManager()
    #
    #     with tf.Session() as s:
    #         s.run(tf.global_variables_initializer())
    #         r = RNTN()
    #         r._get_vocabulary()
    #         r.label_size_ = 5
    #         t = r.get_word(-1)
    #         assert t is not None
