# -*- coding: utf-8 -*-

#
# Tests for training and evaluation of RNTN models.
#

import numpy as np
import random
# from sklearn.utils.estimator_checks import check_estimator
#from src.models import train_model
from src.models.rntn import RNTN
from src.models.data_manager import DataManager
import tensorflow as tf


class TestRNTN(object):

    #def __init__(self):
    #    self.data_mgr = DataManager()

    # This test is failing type checks due to tree data structure. Enable once fixed.
    #def test_rntn_estimator(self):
    #   check_estimator(train_model.RNTN)

    def test_construction(self):
       r = RNTN(model_name='test')
       assert r is not None

    def test_fit(self):
        data_mgr = DataManager()
        x = data_mgr.x_train[0:100]
        r = RNTN(model_name='test', num_epochs=2)
        r.fit(x, None)

    def test_predict(self):
        data_mgr = DataManager()
        r = RNTN(model_name='test')
        x = data_mgr.x_test[0:10]
        y_pred = r.predict(x)
        assert y_pred.shape == (10,)
        print(y_pred)

    def test_predict_proba(self):
        data_mgr = DataManager()
        r = RNTN(model_name='test')
        x = data_mgr.x_test[0:10]
        y_pred = r.predict_proba(x)
        assert y_pred.shape == (10, 5)
        print(y_pred)

    def test_loss(self):
        y_pred = [random.randint(0, 4) for _ in range(10)]
        y_actual = [random.randint(0, 4) for _ in range(10)]
        r = RNTN(model_name='test')
        loss = r._loss(y_pred, y_actual)
        assert np.isfinite(loss)
        print(loss)

    def test_word(self):
        data_mgr = DataManager()

        with tf.Session() as s:
            r = RNTN(model_name='word-test')
            x = data_mgr.x_train
            r._build_vocabulary(x)
            r.label_size_ = 5
            r._build_model_graph_var(r.embedding_size, r.V_, r.label_size_)
            s.run(tf.global_variables_initializer())
            t = r.get_word(23)
            assert t is not None
            assert t.shape == [r.embedding_size, 1]

    def test_word_missing(self):
        data_mgr = DataManager()

        with tf.Session() as s:
            s.run(tf.global_variables_initializer())
            r = RNTN(model_name='word-test')
            x = data_mgr.x_train
            r._build_vocabulary(x)
            r.label_size_ = 5
            t = r.get_word(-1)
            assert t is not None
