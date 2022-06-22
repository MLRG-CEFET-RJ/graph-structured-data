import numpy as np
import os
import pandas as pd
import torch.nn as nn
from ModelInterface import ModelInterface
import torch.optim as optim
from Helper import Helper
import argparse
import matplotlib.pyplot as plt
import time
import torch
from torch.utils.data import DataLoader
import networkx as nx

def get_default_device():
  """Pick GPU if available, else CPU"""
  if torch.cuda.is_available():
      return torch.device('cuda')
  else:
      return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

class NeuralNetwork(nn.Module):
  def __init__(self, NUMBER_NODES):
      super(NeuralNetwork, self).__init__()
      self.fc1 = nn.Linear(((NUMBER_NODES * NUMBER_NODES - NUMBER_NODES) // 2 ), 128)
      self.fc2 = nn.Linear(128, NUMBER_NODES)

  def forward(self, x):
      x = torch.relu(self.fc1(x))
      x = self.fc2(x)
      return x

class CustomLoss(torch.nn.Module):
    
    def __init__(self):
        super(CustomLoss,self).__init__()
        self.mse = nn.MSELoss()

    def loss_repeated_labels(self, roundedOutput):
        batch_size = roundedOutput.shape[0]

        used_labels, counts = torch.unique(roundedOutput, return_counts=True)
        counts = counts.type(torch.FloatTensor)

        counts_shape = counts.shape[0]

        optimalCounts = torch.ones(counts_shape) * batch_size

        return self.mse(counts, optimalCounts)

    def forward(self, output, target):
      loss_mse = self.mse(output, target)

      roundedOutput = output.round()
      loss_repeated = self.loss_repeated_labels(roundedOutput)
      return loss_mse + loss_repeated

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, NUMBER_NODES, patience=7, verbose=False, delta=0, ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.NUMBER_NODES = NUMBER_NODES

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join('saved_models', f'PytorchNeuralNetwork_{self.NUMBER_NODES}_vertices.pt')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

class PytorchNeuralNetwork(ModelInterface):
  def __init__(self, NUMBER_NODES, batch_size, epochs):
    self.NUMBER_NODES = NUMBER_NODES
    self.features_length = (self.NUMBER_NODES * self.NUMBER_NODES - self.NUMBER_NODES) // 2
    self.batch_size = batch_size
    self.epochs = epochs
    self.criterion = CustomLoss()

  def load_train_data(self):
    train_df = pd.read_csv(os.path.join('..', 'datasets', f'dataset_{self.NUMBER_NODES}_train.csv'))
    val_df = pd.read_csv(os.path.join('..', 'datasets', f'dataset_{self.NUMBER_NODES}_val.csv'))

    train_df = pd.concat((train_df, val_df))

    def get_tuple_tensor_dataset(row):
        X = row[0 : self.features_length].astype('float32')
        Y = row[self.features_length + 1: ].astype('float32') # Pula a banda otima na posicao 0
        return torch.from_numpy(X), torch.from_numpy(Y)

    train_dataset = list(map(get_tuple_tensor_dataset, train_df.to_numpy()))
    return train_dataset

  def load_test_data(self):
    test_df = pd.read_csv(os.path.join('..', 'datasets', f'dataset_{self.NUMBER_NODES}_test.csv'))

    def get_tuple_tensor_dataset(row):
        X = row[0 : self.features_length].astype('float32')
        Y = row[self.features_length + 1: ].astype('float32') # Pula a banda otima na posicao 0
        return torch.from_numpy(X), torch.from_numpy(Y)

    test_dataset = list(map(get_tuple_tensor_dataset, test_df.to_numpy()))
    return test_dataset

  def train(self, dataloader, model, optimizer, epoch):
    model.train() # turn on possible layers/parts specific for training, like Dropouts for example
    train_loss = 0
    for X, y in dataloader:
        pred = model(X)
        loss = self.criterion(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return (train_loss / len(dataloader))

  def validate(self, dataloader, model):
    model.eval() # turn off possible layers/parts specific for training, like Dropouts for example
    eval_loss = 0
    with torch.no_grad(): # turn off gradients computation
      for x, y in dataloader:
          pred = model(x)

          loss = self.criterion(pred, y)

          eval_loss += loss.item()
    return (eval_loss / len(dataloader))

  def fit(self):
    BATCH_SIZE =  self.batch_size
    train_data = self.load_train_data()
    test_data = self.load_test_data()

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    self.device = get_default_device()
    print(f"Using {self.device} device")

    train_data = None
    test_data = None

    train_dataloader = DeviceDataLoader(train_dataloader, self.device)
    test_dataloader = DeviceDataLoader(test_dataloader, self.device)

    model = NeuralNetwork(self.NUMBER_NODES).to(self.device)
    optimizer = optim.SGD(model.parameters(), lr=0.0001)

    list_train_loss = []
    list_val_loss = []

    early_stopping = EarlyStopping(NUMBER_NODES=self.NUMBER_NODES, patience=500, verbose=True)

    if not os.path.exists('saved_models'):
      os.makedirs('saved_models')

    for epoch in range(self.epochs):
      # for each epoch, we got a training loss and a validating loss.
      train_loss = self.train(train_dataloader, model, optimizer, epoch)
      list_train_loss.append(train_loss)
      val_epoch_loss = self.validate(test_dataloader, model)
      list_val_loss.append(val_epoch_loss)
      print(f'Epoch {epoch + 1}, train_loss: {train_loss}, val_loss: {val_epoch_loss}')

      # early_stopping needs the validation loss to check if it has decresed, 
      # and if it has, it will make a checkpoint of the current model
      valid_loss = np.average(list_val_loss)
      early_stopping(valid_loss, model)
      if early_stopping.early_stop:
          print("Early stopping")
          break

    train_dataloader = None
    test_dataloader = None

    if not os.path.exists('plotted_figures'):
      os.makedirs('plotted_figures')

    plt.plot(list_train_loss, label='Training Loss')
    plt.plot(list_val_loss, label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join('plotted_figures', f'PytorchNeuralNetwork_loss_{self.NUMBER_NODES}_vertices.jpg'))
    plt.clf()

  def predict(self):
    try:
      helper = Helper(self.NUMBER_NODES)
      self.device = get_default_device()
      print(f"Using {self.device} device")

      model = NeuralNetwork(self.NUMBER_NODES).to(self.device)
      path = os.path.join('saved_models', f'PytorchNeuralNetwork_{self.NUMBER_NODES}_vertices.pt')
      model.load_state_dict(torch.load(path))

      BATCH_SIZE =  self.batch_size
      test_data = self.load_test_data()
      test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
      test_data = None
      test_dataloader = DeviceDataLoader(test_dataloader, self.device)

      sumTest_original = []
      sumTest_pred = []
      sumTest_true = []
      prediction_times = []

      count = 0
      cases_with_repetition = 0

      model.eval()
      with torch.no_grad():
        for x, y in test_dataloader:
          # x = x.cpu()
          # y = y.cpu()
          for features, target in zip(x, y):
            start_time = time.time()

            preds = model(features.unsqueeze(dim=0))

            preds = preds.cpu()

            preds, quantity_repeated, cases_repeated = helper.get_valid_preds(preds.numpy())
            count += quantity_repeated
            cases_with_repetition += cases_repeated

            prediction_times.append(time.time() - start_time)

            features = features.cpu().numpy()
            target = target.cpu().numpy()

            graph = helper.getGraph(features)
            graph = nx.Graph(graph)

            original_band = helper.get_bandwidth(graph, np.array(None))
            sumTest_original.append(original_band)

            pred_band = helper.get_bandwidth(graph, preds[0])
            sumTest_pred.append(pred_band)

            true_band = helper.get_bandwidth(graph, target)
            sumTest_true.append(true_band)

      test_length = 0
      for x, y in test_dataloader:
        test_length += y.shape[0]
      print(test_length)
      test_dataloader = None

      PytorchNeuralNetworkResult = helper.getResult(
        model_name='PytorchNeuralNetwork',
        sumTest_original=sumTest_original,
        sumTest_pred=sumTest_pred,
        sumTest_true=sumTest_true,
        count=count,
        cases_with_repetition=cases_with_repetition,
        prediction_times=prediction_times
      )
      
      return PytorchNeuralNetworkResult
    except FileNotFoundError as e:
      print(e)
      print('Error loading the model from disk, should run model.fit')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Pytorch - Deep learning custom loss')
  parser.add_argument('-v','--vertices', help='number of vertices dataset [7, 9]', required=True)
  parser.add_argument('-m','--mode', help='0 - fit, 1 - predict', required=True)
  parser.add_argument('-b','--batch', help="batch_size - 32, 64, 128, 256, ...", required=True)
  parser.add_argument('-e','--epochs', help="epochs - 10000, 20000, ...", required=True)
  args = parser.parse_args()

  NUMBER_NODES = int(args.vertices)
  batch_size = int(args.batch)
  epochs = int(args.epochs)

  pytorchNeuralNetwork = PytorchNeuralNetwork(NUMBER_NODES=NUMBER_NODES, batch_size=batch_size, epochs=epochs)

  if args.mode == '0':
    pytorchNeuralNetwork.fit()
  if args.mode == '1':
    df_result = pytorchNeuralNetwork.predict()
    print(df_result.to_latex())
