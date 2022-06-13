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
import networkx as nx

class DecisionTreeClassifier(ModelInterface):
  def __init__(self, NUMBER_NODES):
    self.NUMBER_NODES = NUMBER_NODES

  def fit(self):
    model = tree.DecisionTreeClassifier()

    x_train, y_train = super().load_train_data(datatype='int32', NUMBER_NODES=self.NUMBER_NODES)

    model = model.fit(x_train, y_train)

    if not os.path.exists('saved_models'):
      os.makedirs('saved_models')

    dump(model, os.path.join('saved_models', f'DecisionTreeClassifier_{self.NUMBER_NODES}_vertices.joblib'))
  def predict(self):
    try:
      model = load(os.path.join('saved_models', f'DecisionTreeClassifier_{self.NUMBER_NODES}_vertices.joblib')) 

      x_test, y_test = super().load_test_data(datatype='int32', NUMBER_NODES=self.NUMBER_NODES)

      sumTest_original = []
      sumTest_pred = []
      sumTest_true = []
      prediction_times = []

      count = 0
      cases_with_repetition = 0

      helper = Helper(self.NUMBER_NODES)

      test_length = x_test.shape[0]

      for i in range(test_length):
        start_time = time.time()

        output, = model.predict(np.array([x_test[i]]))

        quantity_repeated = helper.count_repeats(np.round(output))

        if quantity_repeated != 0:
            cases_with_repetition += 1
        count += quantity_repeated

        output = helper.get_valid_pred(output)

        prediction_times.append(time.time() - start_time)

        graph = helper.getGraph(x_test[i])
        graph = nx.Graph(graph)

        original_band = helper.get_bandwidth(graph, np.array(None))
        sumTest_original.append(original_band)

        pred_band = helper.get_bandwidth(graph, output)
        sumTest_pred.append(pred_band)

        true_band = helper.get_bandwidth(graph, y_test[i])
        sumTest_true.append(true_band)

      print(test_length)

      DecisionTreeClassifierResult = helper.getResult(
        model_name='DecisionTreeClassifier',
        sumTest_original=sumTest_original,
        sumTest_pred=sumTest_pred,
        sumTest_true=sumTest_true,
        count=count,
        cases_with_repetition=cases_with_repetition,
        prediction_times=prediction_times
      )
      return DecisionTreeClassifierResult
    except FileNotFoundError as e:
      print(e)
      print('Error loading the model from disk, should run model.fit')

    

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Sklearn Decision Tree')
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
