import numpy as np
import os
import pandas as pd
import networkx as nx
from ModelInterface import ModelInterface
import argparse

class DecisionTreeClassifier(ModelInterface):
  def __init__(self, NUMBER_NODES):
    self.NUMBER_NODES = NUMBER_NODES

  def load_data(self):
    train_df = pd.read_csv(os.path.join('datasets', f'dataset_{self.NUMBER_NODES}_train.csv'))
    val_df = pd.read_csv(os.path.join('datasets', f'dataset_{self.NUMBER_NODES}_val.csv'))
    test_df = pd.read_csv(os.path.join('datasets', f'dataset_{self.NUMBER_NODES}_test.csv'))

    featuresNumber = (self.NUMBER_NODES * self.NUMBER_NODES - self.NUMBER_NODES) // 2 
    def get_tuple_tensor_dataset(row):
        X = row[0 : featuresNumber].astype('float32')
        Y = row[featuresNumber + 1: ].astype('float32') # Pula a banda otima na posicao 0
        return X, Y

    train_df = pd.concat((train_df, val_df))

    x_train, y_train = list(map(get_tuple_tensor_dataset, train_df.to_numpy()))
    x_test, y_test = list(map(get_tuple_tensor_dataset, test_df.to_numpy()))
    return x_train, y_train, x_test, y_test

  def fit(self):
    print('fitting')
  def predict(self, x_test):
    for x in x_test:
      print('a predict')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Pytorch - Deep learning custom loss')
  parser.add_argument('-v','--vertices', help='number of vertices dataset [7, 9]', required=True)
  parser.add_argument('-n','--name', help='run custom name [ex.: 7verticesrun]', required=True)
  parser.add_argument('-b','--batch', help='batch size', required=True)
  parser.add_argument('-m','--mode', help=r'{0: fit, 1: predict}', required=True)
  args = parser.parse_args()

  NUMBER_NODES = int(args.vertices)

  decisionTreeClassifier = DecisionTreeClassifier(NUMBER_NODES=NUMBER_NODES)

  if args.mode == '0':
    decisionTreeClassifier.fit()
  if args.mode == '1':
    decisionTreeClassifier.fit()
