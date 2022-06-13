import numpy as np
import os
import pandas as pd
import networkx as nx
from Helper import Helper
import argparse
import time
import networkx as nx
from networkx.utils import reverse_cuthill_mckee_ordering

class ReverseCuthillMckee():
  def __init__(self, NUMBER_NODES):
    self.NUMBER_NODES = NUMBER_NODES

  def load_test_data(self, datatype, NUMBER_NODES):
    features_length = (NUMBER_NODES * NUMBER_NODES - NUMBER_NODES) // 2

    test_df = pd.read_csv(os.path.join('..', 'datasets', f'dataset_{NUMBER_NODES}_test.csv'))

    def get_tuple_tensor_dataset(row):
        X = row[0 : features_length].astype(datatype)
        Y = row[features_length + 1: ].astype(datatype) # Pula a banda otima na posicao 0
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

  def predict(self):
    model = reverse_cuthill_mckee_ordering

    x_test, y_test = self.load_test_data(datatype='int32', NUMBER_NODES=self.NUMBER_NODES)

    sumTest_original = []
    sumTest_pred = []
    sumTest_true = []
    prediction_times = []

    count = 0
    cases_with_repetition = 0

    helper = Helper(self.NUMBER_NODES)

    for i in range(len(x_test)):
      start_time = time.time()

      graph = helper.getGraph(x_test[i])
      graph = nx.Graph(graph)
      output = np.array(list(model(graph)))

      prediction_times.append(time.time() - start_time)

      # quantity_repeated = helper.count_repeats(np.round(output))
      # if quantity_repeated != 0:
      #     cases_with_repetition += 1
      # count += quantity_repeated
      # output = helper.get_valid_pred(output)

      original_band = helper.get_bandwidth(graph, np.array(None))
      sumTest_original.append(original_band)

      pred_band = helper.get_bandwidth(graph, output)
      sumTest_pred.append(pred_band)

      true_band = helper.get_bandwidth(graph, y_test[i])
      sumTest_true.append(true_band)

    test_length = x_test.shape[0]
    print(test_length)

    ReverseCuthillMckeeResult = helper.getResult(
      model_name='ReverseCuthillMckee',
      sumTest_original=sumTest_original,
      sumTest_pred=sumTest_pred,
      sumTest_true=sumTest_true,
      count=count,
      cases_with_repetition=cases_with_repetition,
      prediction_times=prediction_times
    )
    return ReverseCuthillMckeeResult

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Reverse Cuthill Mckee NX Algorithm')
  parser.add_argument('-v','--vertices', help='number of vertices dataset [7, 9]', required=True)
  parser.add_argument('-m','--mode', help=r'{1: predict}', required=True)
  args = parser.parse_args()

  NUMBER_NODES = int(args.vertices)

  reverseCuthillMckee = ReverseCuthillMckee(NUMBER_NODES=NUMBER_NODES)

  if args.mode == '1':
    df_result = reverseCuthillMckee.predict()
    print(df_result.to_latex())
