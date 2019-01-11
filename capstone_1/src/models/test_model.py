# -*- coding: utf-8 -*-

#
# Tests for training and evaluation of all models.
#

# from sklearn.utils.estimator_checks import check_estimator
from src.models import train_model
from src.models.predict_model import predict_model, convert_text_tree


class TestModel(object):
    # This test is failing type checks due to tree data structure. Enable once fixed.
    # def test_rntn_estimator(self):
    #    check_estimator(train_model.RNTN)

    def test_rntn(self):
        params = {'batch_size': [35], 'num_epochs': [3], 'training_rate': [0.01]}
        train_model.train_rntn(model_name='test-rntn', num_samples=100, params=params)

    def test_predict(self):
        y = predict_model('Effective but too-tepid biopic', model_name='test')
        assert y in range(0, 5)

    def test_convert_text_tree(self):
        x = 'But he somehow pulls it off .'
        y = convert_text_tree(x)
        assert y == '(2 (2 But) (2 (2 he) (2 (2 somehow) (2 (2 (2 pulls) (2 (2 it) (2 off))) (2 .)))))'

    #def test_rnn(self):
    #    pass

    #def test_mv_rnn(self):
    #    pass
