import argparse

import pandas as pd

from PytorchNeuralNetwork import PytorchNeuralNetwork
from CatBoostRegressor import CatBoostRegressor
from AdjMatrixCNN import AdjMatrixCNN
from DecisionTreeClassifier import DecisionTreeClassifier
from PointerNetwork import PointerNetwork

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Pytorch - Train and Fit models')
  parser.add_argument('-v','--vertices', help='number of vertices dataset [7, 9]', required=True)
  parser.add_argument('-m','--mode', help='0 - fit, 1 - predict', required=True)

  # PytorchNeuralNetwork config
  parser.add_argument('-bnn','--batch_nn', help="NeuralNetwork batch_size - 32, 64, 128, 256, ...", required=True)
  parser.add_argument('-enn','--epochs_nn', help="NeuralNetwork epochs - 10000, 20000, ...", required=True)

  # AdjMatrixCNN config
  parser.add_argument('-bcnn','--batch_cnn', help="CNN batch_size - 32, 64, 128, 256, ...", required=True)
  parser.add_argument('-ecnn','--epochs_cnn', help="CNN epochs - 200, 256, ...", required=True)

  # PointerNetwork config
  parser.add_argument('-bpnet','--batch_pnet', help="PointerNet batch_size - 32, 64, 128, 256, ...", required=True)
  parser.add_argument('-epnet','--epochs_pnet', help="PointerNet epochs - 52, 64, 128, ...", required=True)
  args = parser.parse_args()

  NUMBER_NODES = int(args.vertices)

  batch_nn = int(args.batch_nn)
  epochs_nn = int(args.epochs_nn)

  batch_cnn = int(args.batch_cnn)
  epochs_cnn = int(args.epochs_cnn)

  batch_pnet = int(args.batch_pnet)
  epochs_pnet = int(args.epochs_pnet)

  pytorchNeuralNetwork = PytorchNeuralNetwork(NUMBER_NODES=NUMBER_NODES, batch_size=batch_nn, epochs=epochs_nn)
  adjMatrixCNN = AdjMatrixCNN(NUMBER_NODES=NUMBER_NODES, batch_size=batch_cnn, epochs=epochs_cnn)
  decisionTreeClassifier = DecisionTreeClassifier(NUMBER_NODES=NUMBER_NODES)
  catBoostRegressor = CatBoostRegressor(NUMBER_NODES=NUMBER_NODES)
  pointerNetwork = PointerNetwork(NUMBER_NODES=NUMBER_NODES, batch_size=batch_pnet, epochs=epochs_pnet)
  
  if args.mode == '0':
    pytorchNeuralNetwork.fit()
    adjMatrixCNN.fit()
    decisionTreeClassifier.fit()
    catBoostRegressor.fit()
    pointerNetwork.fit()
    
  if args.mode == '1':
    pytorchNeuralNetwork_result = pytorchNeuralNetwork.predict()
    adjMatrixCNN_result = adjMatrixCNN.predict()
    decisionTreeClassifier_result = decisionTreeClassifier.predict()
    catBoostRegressor_result = catBoostRegressor.predict()
    pointerNetwork_result = pointerNetwork.predict()
    
    result = pd.concat((
      pytorchNeuralNetwork_result,
      adjMatrixCNN_result,
      decisionTreeClassifier_result,
      catBoostRegressor_result,
      pointerNetwork_result
    ))
    
    print(result.to_latex())