import collections

class RegressionTreeNode(object):
    def __init__(self, j=None, s=None, left_descendent=None, right_descendent=None, const=None):
        self.j = j
        self.s = s
        self.left_descendent = left_descendent
        self.right_descendent = right_descendent
        self.const = const

    def make_terminal(self, c):
        self.left_descendent = None
        self.right_descendent = None
        self.const = c

    def split(self, j, s):
        self.j = j
        self.s = s
        self.left_descendent = RegressionTreeNode()
        self.right_descendent = RegressionTreeNode()

    def get_value(self):
        return self.const

    def print_sub_tree(self, list_of_features_names):
        if self.right_descendent is not None:
            print('if x[{}] <= {}, then: \n return {}'.format(list_of_features_names[self.j], self.s, self.right_descendent.get_const()))
        if self.left_descendent is not None:
            print('if x[{}] > {}, then: \n return {}'.format(list_of_features_names[self.j], self.s, self.left_descendent.get_const()))

    def is_leaf(self):
        if self.left_descendent is None and self.right_descendent is None:
            return True
        else:
            return False

class RegressionTree(object):
    def __init__(self, root):
        self._root = root

    def get_root(self):
        return self._root

    def evaluate(self, x):
        return self.recursion_over_tree(x, self._root)

    def recursion_over_tree(self, x, node: RegressionTreeNode):
        if node.is_leaf():
            return node.get_value()
        if x[node.j] < node.s:
            self.recursion_over_tree(x, node.left_descendent)
        self.recursion_over_tree(x, node.right_descendent)

class RegressionTreeEnsemble(object):
    def __init__(self, trees=None, weights=None, M=None, c=None):
        self.trees = collections.OrderedDict if trees is None else trees
        self.weights = collections.OrderedDict if weights is None else weights
        self.M = 0 if M is None else M
        self.c = 0 if c is None else c

    def add_tree(self, tree, weight):
        self.trees[self.M] = tree
        self.trees[self.M] = weight
        self.M += 1

    def set_initial_constant(self, c):
        self.c = c

    def evaluate(self, x, m):
        sum = 0
        for i, tree in enumerate(self.trees[:m]):
            sum += self.weights[i] * tree.evaluate()
        return sum






