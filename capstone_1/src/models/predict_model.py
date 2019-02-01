# -*- coding: utf-8 -*-

#
# Functionality to predict by using a trained model
# Needs the stanford Core nlp jars and server to be started by
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
#

import json
import logging
import numpy as np
from nltk.parse.corenlp import CoreNLPParser, Tree as nltk_tree
from src.models.rntn import RNTN
from src.features.tree import Tree as features_tree


def predict_model(x, model_name='RNTN_30_tanh_35_5_None_10_0.01_0.001_8544'):
    """ Predict model based on input value.

    :param x:
        A single review text string. Can be multiple sentences.
    :param model_name:
        Trained model name (should be present in models folder)
    :return:
        Sentiment label for the text.
    """
    logging.info('Parsing text {0}'.format(x))

    # Convert string to tree
    x_s = x.strip()
    tree_txt = convert_text_tree(x_s)

    logging.info('Tree structure encoded as {0}'.format(tree_txt))

    tree = features_tree(tree_txt)
    trees = np.asarray([tree]).reshape(-1, 1)

    # Get predictions
    r = RNTN(model_name=model_name, num_epochs=2)
    y_pred = r.predict_proba_full_tree_notf(trees)
    y = np.argmax(y_pred[-1])
    logging.info('probabilities: {0}'.format(y_pred))

    tree_txt = _update_tree_txt(tree_txt, y_pred)
    logging.info('Updated Tree structure json {0}'.format(tree_txt))

    return y, tree_txt


def convert_text_tree(sentence):
    """ Converts a given sentence into a sentiment treebank like tree.

    :param sentence:
        String that needs to be converted.
    :return:
        String encoding tree structure.
    """
    parser = CoreNLPParser()

    # Parse sentence in nltk tree nodes
    root, = next(parser.raw_parse(sentence))

    # Recursively build text
    return get_node_text(root)


def get_node_text(t):
    """ Uses corenlp constituency parser to build a tree structure.

    :param t:
        nltk.tree.Tree instance.
    :return:
        Sentiment Treebank like String compatible with features.Tree.
    """
    logging.debug('Processing node {0}'.format(repr(t)))

    if not isinstance(t, nltk_tree):
        return '(2 {0})'.format(t)

    leaves = t.leaves()

    logging.debug('Found {0} leaves.'.format(len(leaves)))

    assert len(leaves) > 0

    if len(leaves) == 1:
        return '(2 {0})'.format(leaves[0])
    else:
        if len(leaves) == 2:
            return '(2 (2 {0}) (2 {1}))'.format(leaves[0], leaves[1])
        else:
            children = [i for i in t]
            logging.debug('Found {0} children'.format(len(children)))

            txt_1 = get_node_text(children.pop())
            while children:
                txt_2 = get_node_text(children.pop())
                txt_1 = '(2 {0} {1})'.format(txt_2, txt_1)

            return txt_1


def _update_tree_txt(tree_txt, node_probs):
    """ Updates tree text with labels from prediction and returns a json version.

    :param tree_txt:
        Tree text encoding the tree structure.
    :param node_probs:
        A 2D array containing softmax probabilities for each node.
    :return:
        Json string version of updated tree text string
    """

    # Encode into a tree
    tree = features_tree(tree_txt)

    # Walk the tree
    nodes = []
    stack = [tree.root]
    idx = 0
    while stack:
        node = stack.pop()
        if not node.isLeaf:
            stack.append(node.left)
            stack.append(node.right)
        # Insert at zero or if using append reverse to ensure children come before parent.
        node.probabilities = node_probs[-1-idx].tolist()
        node.label = int(np.argmax(node_probs[-1-idx]))
        nodes.insert(0, node)
        idx += 1

    return json.dumps(nodes[-1].to_json())
