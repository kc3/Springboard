#
# Tests for training and evaluation of all models.
#

from src.models.data_manager import DataManager
from src.models import train_model
from src.models.predict_model import predict_seqtoseq


class TestModel(object):

    def test_get_cornell_data(self):
        d = DataManager()
        x, y, a, b = d.get_cornell_data()
        assert x is not None
        assert y is not None
        assert a is not None
        assert b is not None

    def test_train_seqtoseq(self):
        train_model.train_seqtoseq()

    def test_predict_seqtoseq(self):
        y = predict_seqtoseq('Hi!', model_name='test-seqtoseq')
        print(y)

    def test_train_policy(self):
        pass

    def test_predict_policy(self):
        pass
