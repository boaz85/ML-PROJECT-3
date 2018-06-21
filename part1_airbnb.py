import numpy as np
import pandas as pd

CATEGORICAL_VARS = ['room_type', 'city', 'neighborhood']

# 1.
def load_data(path):
    return pd.read_csv(path).drop('name', axis=1).dropna(subset=['price'])

# 2.
def split_train_test(data):
    permutation = np.random.permutation(len(data))
    shuffled = data.iloc[permutation]
    split_point = int(len(data) * 0.8)
    return shuffled.iloc[:split_point], shuffled[split_point:]

# 3. + 4. + 5.
class TrainData(object):

    def __init__(self, data):
        self._raw_data = data
        self._df = data.drop('price', axis=1)
        self._labels = data['price']
        self._categorical_maps = {}

        for column in self._df.dtypes[self._df.dtypes == 'object'].index:
            self._convert_categorical(column)

        self.imputation_map = self._df.mean()
        self._df.fillna(self.imputation_map, inplace=True)

    def _convert_categorical(self, column):
        mean_by_col = self._raw_data.groupby(column)['price'].mean().sort_values()
        self._categorical_maps[column] = mean_by_col.rank()
        self._df[column] = self._df[column].map(self._categorical_maps[column])

    def get_imputation_map(self):
        return self.imputation_map

    def get_categorical_maps(self):
        return self._categorical_maps

    @property
    def X(self):
        return self._df.values

    @property
    def y(self):
        return self._labels.values

# 6.
class TestData(object):

    def __init__(self, data, imputation_map, categorical_maps):

        if 'price' in data.columns:
            self._labels = data['price']
            self._df = data.drop('price', axis=1)
        else:
            self._df = data.copy()

        for column, cat_map in categorical_maps.iteritems():
            self._df[column] = self._df[column].map(cat_map)

        self._df.fillna(imputation_map, inplace=True)

    @property
    def X(self):
        return self._df.values

    @property
    def y(self):
        if hasattr(self, '_labels'):
            return self._labels.values
