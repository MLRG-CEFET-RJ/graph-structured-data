import networkx as nx
import numpy as np
import pandas as pd
import math

class Helper():
  def __init__(self, MAX_NUMBER_NODES):
    self.MAX_NUMBER_NODES = MAX_NUMBER_NODES

  def count_repeats(self, output, number_nodes):
    counts = np.unique(np.round(output))
    repeated = number_nodes - counts.shape[0]
    return repeated

  def get_valid_pred(self, pred, number_nodes):
    valid = np.ones(number_nodes)
    labels = np.arange(0, number_nodes)
    for i in labels:
        min_value = np.amin(pred)
        min_idx, = np.where(pred == min_value)
        min_idx = min_idx[0]
        pred[min_idx] = 100
        valid[min_idx] = i
    return valid

  def get_valid_target(self, target):
    if 35 not in target:
      return target
      
    valid = []
    for i in target:
      if i != 35:
        valid.append(i)
    return np.array(valid)

  def get_pred(self, output, target):
    quantity_repeated = 0
    cases_repeated = 0

    if 35 in target:
      seq_len = np.where(target == 35)[0][0]
    else:
      seq_len = self.MAX_NUMBER_NODES
    
    output = output[ : seq_len]
    repeated = self.count_repeats(output, seq_len)
    pred = self.get_valid_pred(output, seq_len)
    
    if repeated != 0:
      cases_repeated = 1
    quantity_repeated = repeated
    return pred, quantity_repeated, cases_repeated

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

  def solve_quadratic_equation(self, seq_len_features):
    a = 1
    b = -1
    c = seq_len_features * -2

    d = b*b-4*a*c
    sqrt_d = math.sqrt(abs(d))

    sol1 = abs((-b + sqrt_d) // 2*a)
    sol2 = abs((-b - sqrt_d) // 2*a)

    proof1 = (sol1*sol1-sol1)//2

    seq_len = sol1 if seq_len_features == proof1 else sol2

    return abs(int(seq_len))

  def getGraph(self, upperTriangleAdjMatrix):
    if 2 in upperTriangleAdjMatrix:
      seq_len_features = np.where(upperTriangleAdjMatrix == 2)[0][0]
    else:
      seq_len_features = (self.MAX_NUMBER_NODES * self.MAX_NUMBER_NODES - self.MAX_NUMBER_NODES) // 2
    
    upperTriangleAdjMatrix = upperTriangleAdjMatrix[ : seq_len_features]

    seq_len = self.solve_quadratic_equation(seq_len_features)
    
    dense_adj = np.zeros((seq_len, seq_len))
    k = 0
    for i in range(seq_len):
        for j in range(seq_len):
            if i == j:
                continue
            elif i < j:
                dense_adj[i][j] = upperTriangleAdjMatrix[k]
                k += 1
            else:
                dense_adj[i][j] = dense_adj[j][i]
    return dense_adj

  def getResult(self, **kwargs):
    result = [
      [
        np.mean(kwargs['original_bandwidths'][5]),
        np.mean(kwargs['pred_bandwidths'][5]),
        np.mean(kwargs['target_bandwidths'][5])
      ],
      [
        np.mean(kwargs['original_bandwidths'][7]),
        np.mean(kwargs['pred_bandwidths'][7]),
        np.mean(kwargs['target_bandwidths'][7]),
      ],
      [
        np.mean(kwargs['original_bandwidths'][9]),
        np.mean(kwargs['pred_bandwidths'][9]),
        np.mean(kwargs['target_bandwidths'][9]),
      ]
    ]

    df_result = pd.DataFrame(
      result,
      index=[
        '5',
        '7',
        '9'
      ],
      columns=[
        'Original band mean',
        'Predicted band mean',
        'Optimal band mean',
      ]
    )

    metadata = [kwargs['count'], kwargs["cases_with_repetition"], np.mean(kwargs["prediction_times"])]

    df_metadata = pd.DataFrame(
      metadata,
      index=['count', 'cases_with_repetition', 'prediction mean time']
    )

    return df_result, df_metadata
