import numpy as np
import matplotlib.pyplot as plt

from part2 import RegressionTreeEnsemble, RegressionTreeNode, RegressionTree

class CART(object):

    def __init__(self, max_depth, min_node_size, num_thresholds):
        self._max_depth = max_depth
        self._min_node_size = min_node_size
        self._percentiles = np.linspace(0, 100, num_thresholds + 1, False, dtype=int)[1:]

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

            values = np.percentile(np.unique(X[:, j]), self._percentiles, interpolation='higher')

            for s in values:
                r_lt = np.where(X[:, j] <= s)[0]
                r_gt = np.where(X[:, j] > s)[0]
                c_lt = np.mean(y[r_lt]) if len(r_lt) else np.NaN
                c_gt = np.mean(y[r_gt]) if len(r_gt) else np.NaN
                loss = np.sum(np.power(y[r_lt] - c_lt, 2)) + np.sum(np.power(y[r_gt] - c_gt, 2))
                if loss < min_so_far:
                    min_so_far = loss
                    j_opt, s_opt = j, s

        return j_opt, s_opt


class GBRT(object):

    def __init__(self, num_of_basis_functions, num_of_leaves, min_node_size, initial_shrinkage, subsampling_factor, num_thresholds):
        self._num_of_basis_functions = num_of_basis_functions
        self._shrinkage = initial_shrinkage
        self._subsampling = subsampling_factor
        self._cart = CART(int(np.log2(num_of_leaves) + 1), min_node_size, num_thresholds)
        self._shrinkage_checkpoints = ((1 - np.logspace(-1, -5, 5, base=2)) * self._num_of_basis_functions).astype(int)

    def _mean_error(self, predictions, labels):
        return np.mean(np.power(labels - predictions, 2))

    def _subsample(self, num_of_samples):
        samples = np.arange(num_of_samples)
        np.random.shuffle(samples)
        return samples[:int(num_of_samples * self._subsampling)]


    def update_live_view(self, iteration, train_errors, test_errors, block=False):

        plt.plot(range(iteration), train_errors, color='green')
        plt.plot(range(iteration), test_errors, color='orange')

        for shrink_split in self._shrinkage_checkpoints[self._shrinkage_checkpoints <= iteration]:
            plt.axvline(x=shrink_split, color='red')

        plt.show(block=block)
        plt.pause(0.01)

    def fit(self, train_set, test_set=None, liveview=False):

        train_X, train_y = train_set.X, train_set.y

        reg_tree_ensemble = RegressionTreeEnsemble()
        reg_tree_ensemble.set_initial_constant(np.mean(train_y))

        train_f_last = np.repeat(reg_tree_ensemble.c, len(train_X))
        train_errors, test_errors = [], []

        shrinkage_factor = self._shrinkage

        for m in range(1, self._num_of_basis_functions):

            samples = self._subsample(len(train_X))

            g_m = -(train_y[samples] - train_f_last[samples])
            tree = self._cart.fit((train_X[samples], g_m))

            phi_of_x = np.array([tree.evaluate(x) for x in train_X])
            beta_m = np.dot(-g_m, phi_of_x[samples]) / np.sum(np.power(phi_of_x[samples], 2))
            reg_tree_ensemble.add_tree(tree, beta_m)

            train_f_last += shrinkage_factor * beta_m * phi_of_x

            if m in self._shrinkage_checkpoints:
                shrinkage_factor /= 2.0
                print 'Shrinkage factor updated to: ', shrinkage_factor

            train_errors.append(self._mean_error(train_f_last, train_y))
            error_str = 'Learners: {:3d} | Train error: {:15.2f} |'.format(m, train_errors[-1])

            if test_set is not None:

                test_predictions = []

                for x_i in test_set.X:
                    test_predictions.append(reg_tree_ensemble.evaluate(x_i))

                test_errors.append(self._mean_error(test_predictions, test_set.y))
                error_str += ' Test error | {:15.2f} |'.format(test_errors[-1])

            if liveview and m % 10 == 0:
                print error_str
                self.update_live_view(m, train_errors, test_errors)

        if liveview:
            self.update_live_view(self._num_of_basis_functions - 1, train_errors, test_errors, block=True)

        return reg_tree_ensemble, train_errors, test_errors
