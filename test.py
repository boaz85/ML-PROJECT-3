import numpy as np

from part1 import load_data, split_train_test, TrainData, TestData
from part3 import GBRT

np.random.seed(1234)

df = load_data('train.csv')
train, test = split_train_test(df)

train = TrainData(train)
test = TestData(test, train.get_imputation_map(), train.get_categorical_maps())

gbrt = GBRT(200, 4, 1, 1.0, 0.4)
gbrt.fit(train, test)