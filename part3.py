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

    def _mean_error(self, predictions, labels):
        return np.mean(np.power(labels - predictions, 2))

    def fit(self, train_set, test_set=None):

        train_X, train_y = train_set.X, train_set.y

        reg_tree_ensemble = RegressionTreeEnsemble()
        reg_tree_ensemble.set_initial_constant(np.mean(train_y))

        train_f_last = np.repeat(reg_tree_ensemble.c, len(train_X))
        train_errors, test_errors = [], []

        for m in range(1, self._num_of_basis_functions):

            g_m = -(train_y - train_f_last)
            tree = self._cart.fit((train_X, g_m))
            #tree.root.print_sub_tree(train_set._df.columns)

            phi_of_x = np.array([tree.evaluate(x) for x in train_X])
            beta_m = np.dot(-g_m, phi_of_x) / np.sum(np.power(phi_of_x, 2))
            reg_tree_ensemble.add_tree(tree, beta_m)

            train_f_last += beta_m * phi_of_x

            train_errors.append(self._mean_error(train_f_last, train_y))
            error_str = 'Learners: {:3d} | Train error: {:.2f} |'.format(m, train_errors[-1])

            if test_set is not None:

                test_predictions = []

                for x_i in test_set.X:
                    test_predictions.append(reg_tree_ensemble.evaluate(x_i))

                test_errors.append(self._mean_error(test_predictions, test_set.y))
                error_str += ' Test error | {:.2f} |'.format(test_errors[-1])

            print error_str

        return reg_tree_ensemble