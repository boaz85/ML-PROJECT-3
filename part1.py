import numpy as np
import pandas as pd

CATEGORICAL_VARS = ['LotFrontage', 'LotArea', 'YearBuilt', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']

# 1.
def load_data(path):
    return pd.read_csv(path).drop('Id', axis=1).dropna(subset=['SalePrice'])

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
        self._df = data.drop('SalePrice', axis=1)
        self._labels = data['SalePrice']
        self._categorical_maps = {}

        for column in self._df.dtypes[self._df.dtypes == 'object'].index:
            self._convert_categorical(column)

        self.imputation_map = self._df.mean()
        self._df.fillna(self.imputation_map, inplace=True)

    def _convert_categorical(self, column):
        mean_by_col = self._raw_data.groupby(column)['SalePrice'].mean().sort_values()
        self._categorical_maps[column] = mean_by_col.rank()
        self._df[column] = self._df[column].map(self._categorical_maps[column])

    def get_imputation_map(self):
        return self.imputation_map

    def get_categorical_maps(self):
        return self._categorical_maps

    @property
    def X(self):
        return self._df

    @property
    def y(self):
        return self._labels

# 6.
class TestData(object):

    def __init__(self, data, imputation_map, categorical_maps):
        self._df = data.drop('SalePrice', axis=1)
        self._labels = data['SalePrice']

        for column, cat_map in categorical_maps.iteritems():
            self._df[column] = self._df[column].map(cat_map)

        self._df.fillna(imputation_map, inplace=True)

    @property
    def X(self):
        return self._df

    @property
    def y(self):
        return self._labels

df = load_data('train.csv')
train, test = split_train_test(df)

train = TrainData(train)
test = TestData(test, train.get_imputation_map(), train.get_categorical_maps())