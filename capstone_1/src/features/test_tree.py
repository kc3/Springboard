# -*- coding: utf-8 -*-

#
# Tests for class to load and save trees in PTB format.
#
import random
import pytest
from src.features import tree


class TestTree(object):
    def test_node(self):
        node = tree.Node(2, 'Word')
        assert node.label == 2
        assert node.word == 'Word'

    def test_node_print(self):
        with pytest.raises(AttributeError, match='isLeaf is false and no children found.'):
            node = tree.Node(2, 'Word')
            print(str(node))

    def test_tree(self):
        tree_string = '(2 (3 (3 Effective) (2 but)) (1 (1 too-tepid) (2 biopic)))'
        t = tree.Tree(tree_string)
        s = str(t)
        assert s == tree_string

    def test_random_tree(self):
        max_rows = {'train': 8544, 'dev': 1101, 'test': 2210}

        for filename, max_row in max_rows.items():
            file_path = './src/data/interim/trainDevTestTrees_PTB/trees/{0}.txt'.format(filename)
            with open(file_path, 'r') as f:
                row = int(random.random() * max_row)
                for i, line in enumerate(f):
                    if i == row:
                        tree_string = line.strip()
                        t = tree.Tree(tree_string)
                        s = str(t)
                        assert s == tree_string, ""
