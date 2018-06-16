import datetime
import json

import numpy as np

import matplotlib
matplotlib.use('TkAgg')

from part1 import load_data, split_train_test, TrainData, TestData
from part3 import GBRT

np.random.seed(1234)


if __name__ == '__main__':

    df = load_data('train.csv')
    train, test = split_train_test(df)

    train = TrainData(train)
    test = TestData(test, train.get_imputation_map(), train.get_categorical_maps())

    with open('config.json', 'r') as f:
        gbrt_config = json.load(f)

    gbrt = GBRT(**gbrt_config)
    time_before = datetime.datetime.now()
    reg_tree_ensemble, train_errors, test_errors = gbrt.fit(train, test, liveview=True)
    time_after = datetime.datetime.now()

    output = ''
    for name, value in gbrt_config.items():
        output += '{}: {}\n'.format(name.title().replace('_', ''), value)

    output += '\n\nTrain set mean squared error: {}\n'.format(train_errors[-1])
    output += 'Test set mean squared error: {}\n'.format(test_errors[-1])
    output += '\nTraining time: {}'.format(time_after - time_before)

    with open('output.txt', 'w') as f:
        f.write(output)