import numpy as np
import pandas as pd
from ModelInterface import ModelInterface
from Helper import Helper
import argparse
import catboost
import time
import os

class CatBoostRegressor(ModelInterface):
  def __init__(self, NUMBER_NODES):
    super().__init__(NUMBER_NODES)
    self.NUMBER_NODES = NUMBER_NODES
    self.features_length = (self.NUMBER_NODES * self.NUMBER_NODES - self.NUMBER_NODES) // 2

  def fit(self):
    model = catboost.CatBoostRegressor(objective='MultiRMSE', verbose=100)

    x_train, y_train = super().load_train_data(datatype='int32')
    x_test, y_test = super().load_test_data(datatype='int32')

    cat_features = list(range(0, x_test.shape[1]))

    model = model.fit(x_train, y_train, eval_set=(x_test, y_test), cat_features=cat_features, plot=True)

    if not os.path.exists('saved_models'):
      os.makedirs('saved_models')

    model.save_model(os.path.join('saved_models', f'CatBoostRegressor_{self.NUMBER_NODES}_vertices'))
  def predict(self):
    try:
      model = catboost.CatBoostRegressor()
      model.load_model(os.path.join('saved_models', f'CatBoostRegressor_{self.NUMBER_NODES}_vertices')) 

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

      CatBoostRegressorResult = helper.getResult(
        model_name='CatBoostRegressor',
        sumTest_original=sumTest_original,
        sumTest_pred=sumTest_pred,
        sumTest_true=sumTest_true,
        count=count,
        cases_with_repetition=cases_with_repetition,
        mean_time=(end_time - start_time) / test_length
      )
      return CatBoostRegressorResult
    except FileNotFoundError as e:
      print(e)
      print('Error loading the model from disk, should run model.fit')

    

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Catboost - Gradient boosting on decision trees')
  parser.add_argument('-v','--vertices', help='number of vertices dataset [7, 9]', required=True)
  parser.add_argument('-m','--mode', help=r'{0: fit, 1: predict}', required=True)
  args = parser.parse_args()

  NUMBER_NODES = int(args.vertices)

  catBoostRegressor = CatBoostRegressor(NUMBER_NODES=NUMBER_NODES)

  if args.mode == '0':
    catBoostRegressor.fit()
  if args.mode == '1':
    df_result = catBoostRegressor.predict()
    print(df_result.to_latex())
