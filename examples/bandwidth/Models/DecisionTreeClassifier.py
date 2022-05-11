import numpy as np
import os
import pandas as pd
import networkx as nx
from ModelInterface import ModelInterface
import argparse
from joblib import dump, load
from sklearn import tree
import time

class Helper():
  def count_repeats(self, output):
    counts = np.unique(np.round(output))
    repeated = NUMBER_NODES - counts.shape[0]
    return repeated

  def get_valid_pred(self, pred):
      valid = np.ones(NUMBER_NODES)
      labels = np.arange(0, NUMBER_NODES)
      for i in labels:
          min_value = np.amin(pred)
          min_idx, = np.where(pred == min_value)
          min_idx = min_idx[0]
          pred[min_idx] = 100
          valid[min_idx] = i
      return valid
      
  def get_bandwidth(self, Graph, nodelist):
      Graph = nx.Graph(Graph)
      if not Graph.edges:
          return 0
      if nodelist.all() != None:
          L = nx.laplacian_matrix(Graph, nodelist=nodelist)
      else:
          L = nx.laplacian_matrix(Graph)
      x, y = np.nonzero(L)
      return (x-y).max()

  def getGraph(self, upperTriangleAdjMatrix):
      dense_adj = np.zeros((NUMBER_NODES, NUMBER_NODES))
      k = 0
      for i in range(NUMBER_NODES):
          for j in range(NUMBER_NODES):
              if i == j:
                  continue
              elif i < j:
                  dense_adj[i][j] = upperTriangleAdjMatrix[k]
                  k += 1
              else:
                  dense_adj[i][j] = dense_adj[j][i]
      return dense_adj

class DecisionTreeClassifier(ModelInterface):
  def __init__(self, NUMBER_NODES):
    self.NUMBER_NODES = NUMBER_NODES
    self.features_length = (self.NUMBER_NODES * self.NUMBER_NODES - self.NUMBER_NODES) // 2

  def load_train_data(self):
    train_df = pd.read_csv(os.path.join('..', 'datasets', f'dataset_{self.NUMBER_NODES}_train.csv'))
    val_df = pd.read_csv(os.path.join('..', 'datasets', f'dataset_{self.NUMBER_NODES}_val.csv'))
    
    train_df = pd.concat((train_df, val_df))

    def get_tuple_tensor_dataset(row):
        X = row[0 : self.features_length].astype('float32')
        Y = row[self.features_length + 1: ].astype('float32') # Pula a banda otima na posicao 0
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

  def load_test_data(self):
    test_df = pd.read_csv(os.path.join('..', 'datasets', f'dataset_{self.NUMBER_NODES}_test.csv'))

    def get_tuple_tensor_dataset(row):
        X = row[0 : self.features_length].astype('float32')
        Y = row[self.features_length + 1: ].astype('float32') # Pula a banda otima na posicao 0
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

  def fit(self):
    model = tree.DecisionTreeClassifier()

    x_train, y_train = self.load_train_data()

    model = model.fit(x_train, y_train)
    dump(model, 'DecisionTreeClassifier.joblib') 
  def predict(self):
    try:
      model = load('DecisionTreeClassifier.joblib') 

      x_test, y_test = self.load_test_data()

      pred = model.predict(x_test)

      sumTest_original = []
      sumTest_pred = []
      sumTest_true = []

      count = 0
      cases_with_repetition = 0

      helper = Helper()

      start_time = time.time()
      for i in range(len(pred)):
          output = pred[i]

          quantity_repeated = helper.count_repeats(np.round(output))

          if quantity_repeated != 0:
              cases_with_repetition += 1
          count += quantity_repeated

          output = helper.get_valid_pred(output)

          graph = helper.getGraph(x_test[i])
          original_band = helper.get_bandwidth(graph, np.array(None))
          sumTest_original.append(original_band)

          pred_band = helper.get_bandwidth(graph, output)
          sumTest_pred.append(pred_band)

          true_band = helper.get_bandwidth(graph, y_test[i])
          sumTest_true.append(true_band)
      end_time = time.time()

      test_length = pred.shape[0]
      print(test_length)

      DecisionTreeClassifierResult = np.array([
        [
          f'{np.mean(sumTest_original):.2f}±{np.std(sumTest_original):.2f}',
          f'{np.mean(sumTest_pred):.2f}±{np.std(sumTest_pred):.2f}',
          f'{np.mean(sumTest_true):.2f}±{np.std(sumTest_true):.2f}',
          f'{count}',
          f'{cases_with_repetition}',
          f'{(end_time - start_time) / test_length:.4f}'
        ]
      ])

      df_result = pd.DataFrame(
        DecisionTreeClassifierResult,
        index=['DecisionTreeClassifier'],
        columns=[
          'original bandwidth',
          'predicted bandwidth',
          'optimal bandwidth',
          'Repeated labels',
          'Predictions with at least one repeated label',
          'Predicition mean time'
        ]
      )
      return df_result
    except FileNotFoundError as e:
      print(e)
      print('Error loading the model from disk, should run model.fit')

    

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Pytorch - Deep learning custom loss')
  parser.add_argument('-v','--vertices', help='number of vertices dataset [7, 9]', required=True)
  parser.add_argument('-m','--mode', help=r'{0: fit, 1: predict}', required=True)
  args = parser.parse_args()

  NUMBER_NODES = int(args.vertices)

  decisionTreeClassifier = DecisionTreeClassifier(NUMBER_NODES=NUMBER_NODES)

  if args.mode == '0':
    decisionTreeClassifier.fit()
  if args.mode == '1':
    df_result = decisionTreeClassifier.predict()
    print(df_result.to_latex())
