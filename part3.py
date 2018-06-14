from Queue import Queue

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

    def print_sub_tree(self):
        pass

    @property
    def items(self):
        return self.items

class RegressionTree(object):

    def __init__(self, root):
        self.root = root

    def evaluate(self, x):
        pass

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


def recursive_spawn(node, data, current_depth, min_node_size):

    X, y = data

    if current_depth == 0:
        node.make_terminal(np.mean(y))
        return

    j, s = get_optimal_partition((X, y))

    p_l = X[:, j] <= s
    p_r = X[:, j] > s

    if p_l.sum() >= min_node_size and p_r.sum() >= min_node_size:

        left, right = node.split(j, s)
        recursive_spawn(left, (X[p_l], y[p_l]), current_depth - 1, min_node_size)
        recursive_spawn(right, (X[p_r], y[p_r]), current_depth - 1, min_node_size)

    else:
        node.make_terminal(np.mean(y))


def cart(max_depth, min_node_size, dataset):

    root = RegressionTreeNode()
    recursive_spawn(root, dataset, max_depth - 1, min_node_size)

    return RegressionTree(root)


def get_optimal_partition(P):

    X, y = P
    m, d = X.shape
    j_opt, s_opt = None, None

    min_so_far = np.inf

    for j in range(d):

        for s in np.unique(X[:, j]):
            r_lt = np.where(X[:, j] <= s)[0]
            r_gt = np.where(X[:, j] > s)[0]
            c_lt = np.mean(y[r_lt])
            c_gt = np.mean(y[r_gt])
            loss = np.sum(np.power(y[r_lt] - c_lt, 2)) + np.sum(np.power(y[r_gt] - c_gt, 2))
            if loss < min_so_far:
                min_so_far = loss
                j_opt, s_opt = j, s

    return j_opt, s_opt
