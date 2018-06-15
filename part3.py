import numpy as np

from part2 import RegressionTreeEnsemble, RegressionTreeNode, RegressionTree

class CART(object):

    def __init__(self, max_depth, min_node_size):
        self._max_depth = max_depth
        self._min_node_size = min_node_size

    def fit(self, train_set):

        root = RegressionTreeNode()
        self._recursive_spawn(root, train_set, 1)

        return RegressionTree(root)

    def _recursive_spawn(self, node, data, current_depth):

        X, y = data

        if current_depth == self._max_depth:
            node.make_terminal(np.mean(y))
            return

        j, s = self._get_optimal_partition((X, y))

        p_l = X[:, j] <= s
        p_r = X[:, j] > s

        if p_l.sum() >= self._min_node_size and p_r.sum() >= self._min_node_size:

            left, right = node.split(j, s)
            self._recursive_spawn(left, (X[p_l], y[p_l]), current_depth + 1)
            self._recursive_spawn(right, (X[p_r], y[p_r]), current_depth + 1)

        else:
            node.make_terminal(np.mean(y))

    def _get_optimal_partition(self, P):

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


class GBRT(object):

    def __init__(self, num_of_basis_functions, num_of_leaves, min_node_size):
        self._num_of_basis_functions = num_of_basis_functions
        self._cart = CART(np.log2(num_of_leaves) + 1, min_node_size)

    def fit(self, train_set):

        X, y = train_set.X, train_set.y

        learners = [lambda x: np.mean(y)]
        reg_tree_ensemble = RegressionTreeEnsemble()

        for m in range(1, self._num_of_basis_functions):
            g_m = []

            for i, (x_i, y_i) in enumerate(zip(X, y)):
                g_m.append(-(y_i - learners[m - 1](x_i)))

            g_m = np.array(g_m)
            tree = self._cart.fit((X, g_m))
            phi_of_x = np.array([tree.evaluate(x) for x in X])
            beta_m = np.dot(-g_m, phi_of_x) / np.sum(np.power(phi_of_x, 2))
            reg_tree_ensemble.add_tree(tree, beta_m)
            current_learner = (lambda m: lambda x: learners[m - 1](x) - beta_m * tree.evaluate(x))(m)
            learners.append(current_learner)

        return reg_tree_ensemble