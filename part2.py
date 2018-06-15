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

    def print_sub_tree(self):
        pass

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
        self.trees = []
        self.weights = []
        self.M = 0
        self.c = None

    def add_tree(self, tree, weight):
        pass

    def set_initial_constant(self, c):
        self.c = c

    def evaluate(self, x, m):
        pass