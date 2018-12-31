# -*- coding: utf-8 -*-

#
# Tests for training and evaluation of all models.
#

from src.models import train_model


class TestModel(object):
    def test_rntn(self):
        train_model.train_all_models()

    def test_rnn(self):
        pass

    def test_mv_rnn(self):
        pass
