import datetime
import json

import numpy as np

import matplotlib
from multiprocessing import Pool

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

from part1 import load_data, split_train_test, TrainData, TestData
from part3 import GBRT

np.random.seed(1234)

class Worker(object):

    def __init__(self, base_config, train, test):
        self._base_config = base_config
        self._train = train
        self._test = test

    def __call__(self, config_override):
        job_config = {key: config_override.get(key, value) for key, value in self._base_config.items()}
        gbrt = GBRT(**job_config)
        time_before = datetime.datetime.now()
        _, train_errors, test_errors = gbrt.fit(self._train, self._test)
        time_after = datetime.datetime.now()
        return train_errors[-1], test_errors[-1], time_after - time_before

if __name__ == '__main__':

    df = load_data('train.csv')
    train, test = split_train_test(df)

    train = TrainData(train)
    test = TestData(test, train.get_imputation_map(), train.get_categorical_maps())

    with open('config.json', 'r') as f:
        gbrt_config = json.load(f)

    gbrt = GBRT(**gbrt_config)
    reg_tree_ensemble, train_errors, test_errors = gbrt.fit(train, test)

    fig = plt.figure()
    fig.suptitle('MSE as function of NumberOfBasisFunctions')
    plt.plot(range(1, len(train_errors) + 1), train_errors, color='green', label='Train error')
    plt.plot(range(1, len(train_errors) + 1), test_errors, color='orange', label='Test error')
    plt.xlabel('NumberOfBasisFunctions')
    plt.ylabel('MSE')
    plt.legend()
    plt.show(block=False)
    fig.savefig('Deliverable2_MSE_by_NumberOfBasisFunctions.png')

    with open('light_config.json', 'r') as f:
        light_config = json.load(f)

    worker = Worker(light_config, train, test)
    pool = Pool(8)


    num_of_leaves_values = np.logspace(0, 7, 8, base=2)
    args = [{'num_of_leaves': value} for value in num_of_leaves_values]
    results =  pool.map(worker, args)
    train_errors, test_errors, times = zip(*results)

    fig = plt.figure()
    fig.suptitle('MSE as function of NumOfLeaves')
    plt.plot(['{:.0f}'.format(v) for v in num_of_leaves_values], train_errors, color='green', label='Train', marker='o')
    plt.plot(['{:.0f}'.format(v) for v in num_of_leaves_values], test_errors, color='orange', label='Test', marker='o')
    plt.xlabel('NumOfLeaves')
    plt.ylabel('MSE')
    plt.legend()
    plt.show(block=False)
    fig.savefig('Deliverable2_MSE_by_NumOfLeaves.png')


    subsampling_factor_values = np.logspace(-1, 0, 8)
    args = [{'subsampling_factor': value} for value in subsampling_factor_values]
    results = pool.map(worker, args)
    train_errors, test_errors, times = zip(*results)

    fig = plt.figure()
    fig.suptitle('MSE as function of SubsamplingFactor')
    plt.plot(['{:.2f}'.format(v) for v in subsampling_factor_values], train_errors, color='green', label='Train', marker='o')
    plt.plot(['{:.2f}'.format(v) for v in subsampling_factor_values], test_errors, color='orange', label='Test', marker='o')
    plt.xlabel('SubsamplingFactor')
    plt.ylabel('MSE')
    plt.legend()
    plt.show(block=False)
    fig.savefig('Deliverable2_MSE_by_SubsamplingFactor.png')

    fig = plt.figure()
    fig.suptitle('Train time as function of SubsamplingFactor')
    plt.plot(['{:.2f}'.format(v) for v in subsampling_factor_values], [t.total_seconds() for t in times], color='green', marker='o')
    plt.xlabel('SubsamplingFactor')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.show(block=False)
    fig.savefig('Deliverable3_Time_by_SubsamplingFactor.png')


    num_thresholds_values = np.logspace(np.log10(3), 2, 8).astype(int)
    args = [{'num_thresholds': value} for value in num_thresholds_values]
    results = pool.map(worker, args)
    train_errors, test_errors, times = zip(*results)

    fig = plt.figure()
    fig.suptitle('MSE as function of NumThresholds')
    plt.plot(['{:.0f}'.format(v) for v in num_thresholds_values], train_errors, color='green', label='Train', marker='o')
    plt.plot(['{:.0f}'.format(v) for v in num_thresholds_values], test_errors, color='orange', label='Test', marker='o')
    plt.xlabel('NumThresholds')
    plt.ylabel('MSE')
    plt.legend()
    plt.show(block=False)
    fig.savefig('Deliverable2_MSE_by_NumThresholds.png')

    fig = plt.figure()
    fig.suptitle('Train time as function of NumThresholds')
    plt.plot(['{:.0f}'.format(v) for v in num_thresholds_values], [t.total_seconds() for t in times], color='green', marker='o')
    plt.xlabel('NumThresholds')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.show(block=False)
    fig.savefig('Deliverable3_Time_by_NumThresholds.png')