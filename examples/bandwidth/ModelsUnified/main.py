import argparse

import pandas as pd

from PytorchNeuralNetwork import PytorchNeuralNetwork
from examples.bandwidth.ModelsUnified.PointerNetwork import PointerNetwork

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Pytorch - Train and Fit models')
  parser.add_argument('-mv','--max_vertices', help='maximum number of vertices dataset [9]', required=True)
  parser.add_argument('-m','--mode', help='0 - fit, 1 - predict', required=True)

  # PytorchNeuralNetwork config
  parser.add_argument('-bnn','--batch_nn', help="NeuralNetwork batch_size - 32, 64, 128, 256, ...", required=True)
  parser.add_argument('-enn','--epochs_nn', help="NeuralNetwork epochs - 10000, 20000, ...", required=True)

  # # PointerNetwork config
  parser.add_argument('-bpnet','--batch_pnet', help="PointerNet batch_size - 32, 64, 128, 256, ...", required=True)
  parser.add_argument('-epnet','--epochs_pnet', help="PointerNet epochs - 52, 64, 128, ...", required=True)
  args = parser.parse_args()

  MAX_NUMBER_NODES = int(args.max_vertices)

  batch_nn = int(args.batch_nn)
  epochs_nn = int(args.epochs_nn)

  batch_pnet = int(args.batch_pnet)
  epochs_pnet = int(args.epochs_pnet)

  pytorchNeuralNetwork = PytorchNeuralNetwork(MAX_NUMBER_NODES=MAX_NUMBER_NODES, batch_size=batch_nn, epochs=epochs_nn)

  pointerNetwork = PointerNetwork(MAX_NUMBER_NODES=MAX_NUMBER_NODES, batch_size=batch_pnet, epochs=epochs_pnet)
  
  if args.mode == '0':
    pytorchNeuralNetwork.fit()
    pointerNetwork.fit()
    
  if args.mode == '1':
    print("pytorchNeuralNetwork result:")
    df_result, df_metadata = pytorchNeuralNetwork.predict()
    print(df_result.to_latex())
    print(df_metadata.to_latex())

    print("\n")
    
    print("pointerNetwork result:")
    df_result, df_metadata = pointerNetwork.predict()
    print(df_result.to_latex())
    print(df_metadata.to_latex())
