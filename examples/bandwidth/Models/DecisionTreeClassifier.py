import numpy as np
import os
import pandas as pd
import networkx as nx
from ModelInterface import ModelInterface
from Helper import Helper
import argparse
from joblib import dump, load
from sklearn import tree
import time

class DecisionTreeClassifier(ModelInterface):
  def __init__(self, NUMBER_NODES):
    super().__init__(NUMBER_NODES)
    self.NUMBER_NODES = NUMBER_NODES
    self.features_length = (self.NUMBER_NODES * self.NUMBER_NODES - self.NUMBER_NODES) // 2

  def fit(self):
    model = tree.DecisionTreeClassifier()

    x_train, y_train = super().load_train_data(datatype='int32')

    model = model.fit(x_train, y_train)

    if not os.path.exists('saved_models'):
      os.makedirs('saved_models')

    dump(model, 'saved_models/DecisionTreeClassifier.joblib') 
  def predict(self):
    try:
      model = load('saved_models/DecisionTreeClassifier.joblib') 

      x_test, y_test = super().load_test_data(datatype='int32')

      pred = model.predict(x_test)

      sumTest_original = []
      sumTest_pred = []
      sumTest_true = []

      count = 0
      cases_with_repetition = 0

      helper = Helper(self.NUMBER_NODES)

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
