# by: Jindong Wang, jindongwang
# https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py

# Compute MMD (maximum mean discrepancy) using numpy and scikit-learn.
import json
import os.path
import sys

import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt


def mmd_linear(X, Y):
    """MMD using linear kernel (i.e., k(x,y) = <x,y>)
    Note that this is not the original linear MMD, only the reformulated and faster version.
    The original version is:
        def mmd_linear(X, Y):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
            return XX.mean() + YY.mean() - 2 * XY.mean()

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Returns:
        [scalar] -- [MMD value]
    """
    delta = X.mean(0) - Y.mean(0)
    return delta.dot(delta.T)


def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})

    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def mmd_poly(X, Y, degree=2, gamma=1, coef0=0):
    """MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        degree {int} -- [degree] (default: {2})
        gamma {int} -- [gamma] (default: {1})
        coef0 {int} -- [constant item] (default: {0})

    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
    YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
    XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
    return XX.mean() + YY.mean() - 2 * XY.mean()


if __name__ == '__main__':
    '''
    a = np.arange(1, 10).reshape(3, 3)
    b = [[7, 6, 5], [4, 3, 2], [1, 1, 8], [0, 2, 5]]
    b = np.array(b)
    print('a = {}\n'.format(a))
    print('b = {}\n'.format(b))
    print('mmd linear:  {:2.1f}'.format(mmd_linear(a, b)))              # 6.0
    print('mmd rbf:     {:6.4f}'.format(mmd_rbf(a, b)))                 # 0.5822
    print('mmd poly:    {:6.1f}'.format(mmd_poly(a, b)))                # 2436.5
    '''
    key_case = 'rl'
    drive = 'E:\\'
    myPath_base = os.path.join(drive, 'UTSA')
    temp_path = os.path.join(myPath_base, 'paper3_DM\\paper3_data')
    temp_path = os.path.join(temp_path, 'gb_dm_case_{}_rev1\\gb_dm_1d_case_{}.json'.format(key_case, key_case))

    with open(temp_path, 'r') as f:
        a = np.asarray(json.load(f))

    a = a[:10000, :]
    temp_path = os.path.join(myPath_base, 'paper3_DM\\paper3_data')
    temp_path = os.path.join(temp_path, 'mitbih_64_allN.json')
    with open(temp_path, 'r') as f:
        rl_all = np.asarray(json.load(f))

    rl = rl_all[:10000, :]
    temp_path = os.path.join(myPath_base, 'paper3_DM\\paper3_data\\1_distributions')
    f_name = 'mmd_case_{}_and_rl.txt'.format(key_case)
    path = os.path.join(temp_path, f_name)
    with open(path, 'w') as sys.stdout:
        print('MMD between case {} and real data from training set:\n'.format(key_case))
        print('mmd linear:      {:2.1f}'.format(mmd_linear(a, rl)))
        print('mmd rbf:         {:6.4f}'.format(mmd_rbf(a, rl)))
        print('mmd poly:       {:6.1f}'.format(mmd_poly(a, rl)))
    sys.stdout = sys.__stdout__
    print('Finished ...')
    brk = 'here'
