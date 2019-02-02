# -*- coding: utf-8 -*-

#
# Tests for training and evaluation of RNTN models.
#

import math
import numpy as np
import random
# from sklearn.utils.estimator_checks import check_estimator
from src.features.tree import Tree
from src.models.rntn import RNTN
from src.models.data_manager import DataManager
import tensorflow as tf

class TestRNTN(object):

    # This test is failing type checks due to tree data structure. Enable once fixed.
    #def test_rntn_estimator(self):
    #   check_estimator(train_model.RNTN)

    def test_construction(self):
       r = RNTN(model_name='test')
       assert r is not None

    def test_loss(self):
        data_mgr = DataManager()
        y_pred = [random.randint(0, 4) for _ in range(10)]
        proba = [[random.random() for _ in range(5)] for _ in range(10)]
        proba = proba / np.sum(proba)
        r = RNTN(model_name='test')
        loss = r._loss(y_pred, proba)
        assert np.isfinite(loss)
        print(loss)

    def test_fit(self):
        data_mgr = DataManager()
        x = np.asarray(data_mgr.x_train[0:100]).reshape(-1, 1)
        r = RNTN(model_name='test')
        r.fit(x, None)

    def test_predict(self):
        data_mgr = DataManager()
        r = RNTN(model_name='test')
        x = np.asarray(data_mgr.x_test[0:10]).reshape(-1, 1)
        y_pred = r.predict(x)
        assert y_pred.shape == (10,)
        print(y_pred)

    def test_predict_proba(self):
        data_mgr = DataManager()
        r = RNTN(model_name='test')
        x = np.asarray(data_mgr.x_test[0:10]).reshape(-1, 1)
        y_pred = r.predict_proba(x)
        assert y_pred.shape == (10, 5)
        print(y_pred)

    def test_predict_proba_full_tree(self):
        data_mgr = DataManager()
        r = RNTN(model_name='test')
        x = np.asarray(data_mgr.x_test[0:10]).reshape(-1, 1)
        y_pred = r.predict_proba_full_tree(x)
        print(y_pred)

    def test_word(self):
        data_mgr = DataManager()

        with tf.Session() as s:
            r = RNTN(model_name='word-test')
            x = np.asarray(data_mgr.x_train).reshape(-1, 1)
            r._build_vocabulary(x[:, 0])
            r._build_model_graph_var(r.embedding_size, r.V_, r.label_size,
                                     r._regularization_l2_func(r.regularization_rate))
            s.run(tf.global_variables_initializer())
            t = r.get_word(23)
            assert t is not None
            assert t.shape == [r.embedding_size, 1]

    def test_word_missing(self):
        data_mgr = DataManager()

        with tf.Session() as s:
            s.run(tf.global_variables_initializer())
            r = RNTN(model_name='word-test')
            x = np.asarray(data_mgr.x_train).reshape(-1, 1)
            r._build_vocabulary(x[:, 0])
            t = r.get_word(-1)
            assert t is not None

    def test_over_sampler(self):
        data_mgr = DataManager()

        def foo(node):
            r = np.zeros(5)
            r[node.label] = 1
            if node.isLeaf:
                return r
            else:
                return foo(node.left) + foo(node.right) + r

        x = data_mgr.x_train
        y = np.zeros(5)
        for i in range(len(x)):
            y += foo(x[i].root)
        z = np.max(y)
        print(np.ones(5)*z/(y))

    def test_export_model(self):
        data_mgr = DataManager()
        x = np.asarray(data_mgr.x_train[0:100]).reshape(-1, 1)
        r = RNTN(model_name='test-export')
        r.fit(x, None)

        save_dir = r._get_save_dir()
        L = np.load('{0}/L.npy'.format(save_dir))
        assert L.shape == (r.embedding_size, len(r.vocabulary_))

        W = np.load('{0}/W.npy'.format(save_dir))
        assert W.shape == (r.embedding_size, r.embedding_size*2)

        b = np.load('{0}/b.npy'.format(save_dir))
        assert b.shape == (r.embedding_size, 1)

        U = np.load('{0}/U.npy'.format(save_dir))
        assert U.shape == (r.embedding_size, r.label_size)

        bs = np.load('{0}/bs.npy'.format(save_dir))
        assert bs.shape == (r.label_size, 1)

        T_s = np.load('{0}/T.npy'.format(save_dir))
        T = T_s.reshape(r.embedding_size*2, r.embedding_size*2, r.embedding_size)
        vocab = r.vocabulary_

        word_1 = list(vocab.keys())[0]
        print('Got word: {0}'.format(word_1))
        good_embed = L[:, vocab[word_1]]
        print('Good word: {0}'.format(good_embed))
        print('mul shape: {0}'.format(np.matmul(np.transpose(U), good_embed)))
        print('bs: {0}',format(bs))
        good_class = np.matmul(np.transpose(U), good_embed) + bs.reshape(-1)
        print('Good logit: {0}, class: {1}'.format(good_class, np.argmax(good_class)))

        word_2 = list(vocab.keys())[1]
        print('Got word: {0}'.format(word_2))
        movie_embed = L[:, vocab[word_2]]
        print('Movie word: {0}'.format(movie_embed))
        movie_class = np.matmul(np.transpose(U), movie_embed) + bs.reshape(-1)
        print('Movie logit: {0}, class: {1}'.format(movie_class, np.argmax(movie_class)))

        X = np.concatenate([good_embed, movie_embed])
        print('Combined vector: {0}'.format(X))

        zs = np.matmul(W, X) + b.reshape(-1)
        print('zs: {0}'.format(zs))

        zd = np.zeros([r.embedding_size])
        for i in range(r.embedding_size):
            T_i = T[:, :, i]
            zd[i] = np.matmul(np.matmul(np.transpose(X), T_i), X)
        print('zd: {0}'.format(zd))

        a = np.tanh(zs + zd)
        print('a: {0}'.format(a))
        a_class = np.matmul(np.transpose(U), a) + bs.reshape(-1)
        print('Combined logit: {0}, class: {1}'.format(a_class, np.argmax(a_class)))

    def test_get_weights(self):
        r = RNTN()
        w = r._get_weight_by_height(0, 0)
        assert(w == 124.26106194690266)

        txt = "(2 (3 (3 Effective) (2 but)) (1 (1 too-tepid) (2 biopic)))"
        tree = Tree(txt)
        w = r._get_tree_weights(tree)
        exp_w = [12.184571329399514, 1.0, 5.018175209014904, 18.01347017318794, 1.0, 7.283038776048536, 1.0]
        cmp_w = [math.isclose(w[i], exp_w[i]) for i in range(len(w))]
        assert all(cmp_w)
