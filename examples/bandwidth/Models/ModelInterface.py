from abc import ABC, abstractmethod
import pandas as pd
import os
import numpy as np

class ModelInterface(ABC):
    # default method
    def load_train_data(self, datatype):
      train_df = pd.read_csv(os.path.join('..', 'datasets', f'dataset_{self.NUMBER_NODES}_train.csv'))
      val_df = pd.read_csv(os.path.join('..', 'datasets', f'dataset_{self.NUMBER_NODES}_val.csv'))
      
      train_df = pd.concat((train_df, val_df))

      def get_tuple_tensor_dataset(row):
          X = row[0 : self.features_length].astype(datatype)
          Y = row[self.features_length + 1: ].astype(datatype) # Pula a banda otima na posicao 0
          return X, Y

      train_dataset = list(map(get_tuple_tensor_dataset, train_df.to_numpy()))

      X = []
      Y = []
      for x, y in train_dataset:
          X.append(x)
          Y.append(y)
      x_train = np.array(X)
      y_train = np.array(Y)

      return x_train, y_train

    # default method
    def load_test_data(self, datatype):
      test_df = pd.read_csv(os.path.join('..', 'datasets', f'dataset_{self.NUMBER_NODES}_test.csv'))

      def get_tuple_tensor_dataset(row):
          X = row[0 : self.features_length].astype(datatype)
          Y = row[self.features_length + 1: ].astype(datatype) # Pula a banda otima na posicao 0
          return X, Y

      test_dataset = list(map(get_tuple_tensor_dataset, test_df.to_numpy()))

      X = []
      Y = []
      for x, y in test_dataset:
          X.append(x)
          Y.append(y)
      x_test = np.array(X)
      y_test = np.array(Y)

      return x_test, y_test

    @abstractmethod
    def fit(self):
      pass

    @abstractmethod
    def predict(self, x_test):
      pass