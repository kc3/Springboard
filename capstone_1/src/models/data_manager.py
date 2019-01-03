# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from src.features.tree import Tree

#
# Singleton to avoid multiple copies.
# Class to hold a copy of data
#


class DataManager:
    """Load interim data for trees as strings."""

    _def_path = './src/data/interim/trainDevTestTrees_PTB/trees/'

    __instance = None

    def __new__(cls, *args, **kwargs):
        """Create the object on first instantiation."""

        if cls.__instance is None:
            cls.__instance = object.__new__(cls)

        return cls.__instance

    def __init__(self, path=None, max_rows=None):
        if path is None:
            path = self._def_path

        # Load data and store them as trees
        self.x_train = self._load(self._make_file_name(path, 'train'), max_rows)
        self.x_dev = self._load(self._make_file_name(path, 'dev'), max_rows)
        self.x_test = self._load(self._make_file_name(path, 'test'), max_rows)

        # Build Corpus
        self.countvectorizer = CountVectorizer()
        self._build_corpus()

    @staticmethod
    def _make_file_name(file_path, file_name):
        return '{0}{1}.txt'.format(file_path, file_name)

    @staticmethod
    def _load(file_path, max_rows=None):
        """Loads entire content of the file."""
        s = []
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if max_rows is None or i < max_rows:
                    tree_string = line.strip()
                    t = Tree(tree_string)
                    s.append(t)
                else:
                    break

            return s

    def _build_corpus(self):
        """Builds corpus from tree strings"""
        x = self.x_train + self.x_dev
        corpus = []
        for i in range(len(x)):
            corpus.append(x[i].text())

        # Use CountVectorizer to build dictionary of words.
        self.countvectorizer.fit(corpus)
