import numpy as np
import operator

def tree_relative_feature_importance(tree, data):
    root = tree.root
    features_relative_loss = {}

    recursive_relative_error_eval(root, data.X, data.y, features_relative_loss)
    return sorted(features_relative_loss.items(), key=operator.itemgetter(1))

def recursive_relative_error_eval(node, X, y, features_relative_loss):
    j, s = node.j, node.s
    c_mean = np.mean(y)
    loss_bef = np.sum(np.power(y - c_mean, 2))

    if node.is_terminal():
        return (-1) * loss_bef

    X_lt, y_lt = X[np.where(X[:, j] <= s)[0]], y[np.where(X[:, j] <= s)[0]]
    X_gt, y_gt = X[np.where(X[:, j] > s)[0]], y[np.where(X[:, j] > s)[0]]
    loss_lt = recursive_relative_error_eval(node.left_descendant, X_lt, y_lt, features_relative_loss)
    loss_gt = recursive_relative_error_eval(node.right_descendant, X_gt, y_gt, features_relative_loss)
    loss = np.sum(loss_lt + loss_gt)
    relative_loss = loss - loss_bef

    if j not in features_relative_loss:
        features_relative_loss[j] = relative_loss
    else:
        features_relative_loss[j] += relative_loss

    return loss_bef

def error(node, P):
    j, s = node.j, node.s
    X, y = P

    r_lt = np.where(X[:, j] <= s)[0]
    r_gt = np.where(X[:, j] > s)[0]
    c_lt = np.mean(y[r_lt])
    c_gt = np.mean(y[r_gt])
    loss = np.sum(np.power(y[r_lt] - c_lt, 2)) + np.sum(np.power(y[r_gt] - c_gt, 2))

    return loss


def ensemble_relative_feature_importance(ensemble, data):
    ensemble_importance = {}
    for tree in ensemble.trees:
        tree_importance = tree_relative_feature_importance(tree, data)
        ensemble_importance = merge_dictionaries(ensemble_importance, tree_importance)
    ensemble_importance = {k: v / ensemble.M for k, v in ensemble_importance.iteritems()}
    ensemble_importance = {k: abs(v) / max(map(abs, ensemble_importance.values())) for k, v in ensemble_importance.iteritems()}
    return ensemble_importance

def merge_dictionaries(dict1, dict2):
    dict3 = dict1
    for key in dict(dict2).keys():
        try:
            dict1[key] += dict(dict2)[key]
        except:
            dict3[key] = dict(dict2)[key]
    return dict3

