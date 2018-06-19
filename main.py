import datetime
import json
import sys

import numpy as np

import matplotlib

from part6 import ensemble_relative_feature_importance

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

from part1 import load_data, split_train_test, TrainData, TestData
from part3 import GBRT

np.random.seed(4321)

if __name__ == '__main__':

    df = load_data(sys.argv[1])
    train, test = split_train_test(df)

    train = TrainData(train)
    test = TestData(test, train.get_imputation_map(), train.get_categorical_maps())

    with open('config.json', 'r') as f:
        gbrt_config = json.load(f)


    gbrt = GBRT(**gbrt_config)
    time_before = datetime.datetime.now()
    train_errors, test_errors = gbrt.fit(train, test, liveview=True)
    time_after = datetime.datetime.now()

    output = ''
    for name, value in gbrt_config.items():
        output += '{}: {}\n'.format(name.title().replace('_', ''), value)

    output += '\n\nTrain set mean squared error: {}\n'.format(train_errors[-1])
    output += 'Test set mean squared error: {}\n'.format(test_errors[-1])
    output += '\nOptimal number of learners: {}\n'.format(gbrt.ensemble.M)
    output += 'Train set mean squared error at optimum: {}\n'.format(train_errors[gbrt.ensemble.M - 1])
    output += 'Test set mean squared error at optimum: {}\n'.format(test_errors[gbrt.ensemble.M - 1])
    output += '\nTraining time: {}'.format(time_after - time_before)

    with open('output.txt', 'w') as f:
        f.write(output)

    f = open('Deliverables/HousingDS/Deliverable4_TreesPrint.txt', 'w')
    for i, tree in enumerate(gbrt.ensemble.trees[:5]):
        f.write('Tree #{}\n--------\n\n'.format(i + 1))
        f.write(tree.root.get_string_representation().format(*train.features) + '\n\n\n')
    f.close()

    feature_importance = ensemble_relative_feature_importance(gbrt.ensemble, train)

    fig = plt.figure()
    fig.suptitle('Feature importance')
    feature_indices, importance = zip(*sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:8])

    plt.bar(train.features[list(feature_indices)], importance)
    plt.xticks(rotation=70)
    plt.subplots_adjust(bottom=0.25)
    plt.ylabel('Importance')
    plt.show(block=False)
    fig.savefig('Deliverables/HousingDS/Deliverable4_FeatureImportanceBarGraph.png')