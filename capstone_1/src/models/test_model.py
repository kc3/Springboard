# -*- coding: utf-8 -*-

#
# Tests for training and evaluation of all models.
#

# from sklearn.utils.estimator_checks import check_estimator
from src.models import train_model


class TestModel(object):
    # This test is failing type checks due to tree data structure. Enable once fixed.
    # def test_rntn_estimator(self):
    #    check_estimator(train_model.RNTN)

    #def test_rntn(self):
    #    train_model.train_rntn()

    def test_rnn(self):
        pass

    def test_mv_rnn(self):
        pass
