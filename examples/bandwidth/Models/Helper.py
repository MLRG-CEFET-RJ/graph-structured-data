import networkx as nx
import numpy as np
import pandas as pd

class Helper():
  def __init__(self, NUMBER_NODES):
    self.NUMBER_NODES = NUMBER_NODES

  def count_repeats(self, output):
    counts = np.unique(np.round(output))
    repeated = self.NUMBER_NODES - counts.shape[0]
    return repeated

  def get_valid_pred(self, pred):
    valid = np.ones(self.NUMBER_NODES)
    labels = np.arange(0, self.NUMBER_NODES)
    for i in labels:
        min_value = np.amin(pred)
        min_idx, = np.where(pred == min_value)
        min_idx = min_idx[0]
        pred[min_idx] = 100
        valid[min_idx] = i
    return valid

  def get_valid_preds(self, batch_output):
    quantity_repeated = 0
    cases_repeated = 0
    for i, output in enumerate(batch_output):
      repeated = self.count_repeats(output)
      
      batch_output[i] = self.get_valid_pred(output)
      
      if repeated != 0:
        cases_repeated += 1
      quantity_repeated += repeated
    return batch_output, quantity_repeated, cases_repeated

  def get_bandwidth(self, Graph, nodelist):
    if not Graph.edges:
        return 0
    if nodelist.all() != None:
        L = nx.laplacian_matrix(Graph, nodelist=nodelist)
    else:
        L = nx.laplacian_matrix(Graph)
    x, y = np.nonzero(L)
    L = None
    return (x-y).max()

  def getGraph(self, upperTriangleAdjMatrix):
    dense_adj = np.zeros((self.NUMBER_NODES, self.NUMBER_NODES))
    k = 0
    for i in range(self.NUMBER_NODES):
        for j in range(self.NUMBER_NODES):
            if i == j:
                continue
            elif i < j:
                dense_adj[i][j] = upperTriangleAdjMatrix[k]
                k += 1
            else:
                dense_adj[i][j] = dense_adj[j][i]
    return dense_adj
    
  def getResult(self, model_name, **kwargs):
    result = np.array([
        [
          f'{np.mean(kwargs["sumTest_original"]):.2f}±{np.std(kwargs["sumTest_original"], ddof=1):.2f}',
          f'{np.mean(kwargs["sumTest_pred"]):.2f}±{np.std(kwargs["sumTest_pred"], ddof=1):.2f}',
          f'{np.mean(kwargs["sumTest_true"]):.2f}±{np.std(kwargs["sumTest_true"], ddof=1):.2f}',
          f'{kwargs["count"]}',
          f'{kwargs["cases_with_repetition"]}',
          f'{np.mean(kwargs["prediction_times"]):.4f}±{np.std(kwargs["prediction_times"], ddof=1):.4f}'
        ]
      ])

    df_result = pd.DataFrame(
      result,
      index=[f'{model_name}'],
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
