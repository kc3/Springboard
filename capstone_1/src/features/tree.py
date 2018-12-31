# -*- coding: utf-8 -*-

#
# tree.py
# File to read the PTB tree structures in Stanford Sentiment Treebank
# Files in trainDevTestTrees_PTB/*.txt
#

#
# Class to represent a single Node in the tree.
# Trees are assumed to be binary.
#


class Node:
    def __init__(self, label, word=None):

        # Sentiment Label associated with the Node. (an ordinal in range 1-5)
        self.label = label

        # Indicates whether the node is a Leaf node or an intermediate node in the tree.
        self.isLeaf = False

        # Sentiment Word from the corpus.
        # Populated only if the node is a Leaf node.
        self.word = word

        # Subtrees
        # Populated only if intermediate nodes.
        self.left = None
        self.right = None

    def __str__(self):
        if self.isLeaf:
            return '({0} {1})'.format(self.label, self.word)
        else:
            if self.left is not None and self.right is not None:
                return '({0} {1} {2})'.format(self.label, self.left, self.right)
            else:
                raise AttributeError('isLeaf is false and no children found.')

#
# Class to represent a Tree.
#


class Tree:
    def __init__(self, tree_string):

        # Root of the tree.
        self.root = self._parse(tree_string)

    def __str__(self):
        return '{0}'.format(self.root)

    @staticmethod
    def _parse(tree_string):
        """Parse tree structure from string in PTB format"""

        #
        # Example of the expected string
        # (2 (3 (3 Effective) (2 but)) (1 (1 too-tepid) (2 biopic)))
        #

        # Variable to hold root
        root = None
        stack = []
        idx = 0
        n = len(tree_string)

        while idx < n:

            # Remove whitespace
            while tree_string[idx] == ' ':
                idx += 1

            # If open parenthesis, create a new node
            if tree_string[idx] == '(':

                # Parse sentiment label that immediately follows.
                idx += 1
                assert idx < n and tree_string[idx].isdigit(), "Expected sentiment label after open parenthesis. {0}"\
                    .format(tree_string[idx:])

                label = int(tree_string[idx])
                assert 0 <= label <= 4, "Sentiment label is integer between 0 and 4 inclusive. {0}"\
                    .format(tree_string[idx:])
                idx += 1

                # Create a new node
                new_node = Node(label)

                # If we find another open brace before the next closing brace,
                # This is an intermediate node else it is leaf.
                open_idx = tree_string.find('(', idx)
                close_idx = tree_string.find(')', idx)

                assert close_idx != -1, "Expected closing parenthesis after open parenthesis. {0}"\
                    .format(tree_string[idx:])

                if open_idx == -1 or close_idx < open_idx:
                    # Found a leaf
                    # Read token
                    token = tree_string[idx:close_idx].strip()
                    new_node.isLeaf = True
                    new_node.word = token
                    idx = close_idx
                else:
                    idx += 1

                stack.append(new_node)

            else:
                if tree_string[idx] == ')':
                    assert stack, "Closing parenthesis found before any tokens. {0}"\
                        .format(tree_string[idx:])

                    # Get node
                    node = stack.pop()

                    # Get parent
                    if not stack:
                        # No parent case, update root
                        root = node
                    else:
                        # Add node to parent
                        parent = stack.pop()
                        if parent.left is None:
                            parent.left = node
                        else:
                            assert parent.right is None
                            parent.right = node

                        # Add parent back to stack again
                        stack.append(parent)

                    idx += 1
                else:
                    raise RuntimeError("Parsing error.{0}, {1}".format(tree_string[idx:], stack))

        return root
