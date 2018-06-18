import numpy as np

class RegressionTreeNode(object):

    def __init__(self):
        self.s = None
        self.j = None
        self.left_descendant = None
        self.right_descendant = None
        self.const = None

    def make_terminal(self, c):
        self.const = c

    def split(self, j, s):
        self.j, self.s = j, s
        self.left_descendant = RegressionTreeNode()
        self.right_descendant = RegressionTreeNode()

        return self.left_descendant, self.right_descendant

    def print_sub_tree(self, feature_names):
        print self.get_string_representation().format(*list(feature_names))

    def get_string_representation(self, indent_level=0):
        indent_space = indent_level * '\t'

        if self.is_terminal():
            return '{}return {}\n'.format(indent_space, self.const)

        str = '''{}if x['{{{}}}'] <= {} then\n'''.format(indent_space, self.j, self.s)
        str += self.left_descendant.get_string_representation(indent_level + 1)
        str += '''{}if x['{{{}}}'] > {} then\n'''.format(indent_space, self.j, self.s)
        str += self.right_descendant.get_string_representation(indent_level + 1)

        return  str

    def is_terminal(self):
        return self.left_descendant is None

class RegressionTree(object):

    def __init__(self, root):
        self.root = root

    def _recursive_eval(self, node, x):
        if node.is_terminal():
            return node.const

        if x[node.j] <= node.s:
            return self._recursive_eval(node.left_descendant, x)

        return self._recursive_eval(node.right_descendant, x)

    def evaluate(self, x):
        return self._recursive_eval(self.root, x)


class RegressionTreeEnsemble(object):

    def __init__(self):
        self._trees = []
        self.weights = []
        self.c = None
        self.M = 0

    def add_tree(self, tree, weight):
        self._trees.append(tree)
        self.weights.append(weight)
        self.M += 1

    def set_initial_constant(self, c):
        self.c = c

    def evaluate(self, x, m=None):
        m = m or self.M
        predictions = [tree.evaluate(x) for tree in self._trees[:m]]
        return self.c + np.dot(predictions, self.weights[:m])

    @property
    def trees(self):
        return self._trees[:self.M]