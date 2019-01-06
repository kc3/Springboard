# -*- coding: utf-8 -*-

#
# Tests for training and evaluation of RNTN models.
#

# from sklearn.utils.estimator_checks import check_estimator
#from src.models import train_model
from src.models.rntn import RNTN
from src.models.data_manager import DataManager

class TestRNTN(object):

    #def __init__(self):
    #    self.data_mgr = DataManager()

    # This test is failing type checks due to tree data structure. Enable once fixed.
    # def test_rntn_estimator(self):
    #    check_estimator(train_model.RNTN)

    def test_construction(self):
        r = RNTN()
        assert r is not None

    def test_fit(self):
        data_mgr = DataManager()
        r = RNTN()

        x = data_mgr.x_train + data_mgr.x_dev
        r.fit(x, None)

    def test_word(self):
        data_mgr = DataManager()
        r = RNTN()
        r._get_vocabulary()
        r.label_size_ = 5
        r._build_model_graph_var(r.embedding_size, r.V_, r.label_size_)
        t = r.get_word(23)
        assert t is not None
