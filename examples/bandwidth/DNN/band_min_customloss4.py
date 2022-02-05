# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
import random
import pickle
import networkx as nx
import os

# %%
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# %%
NUMBER_NODES = 7

# %%
os.path.join('datasets', f'dataset_{NUMBER_NODES}_train.csv')

# %%
a = os.path.join('datasets')
os.listdir(a)

# %%
def load_data():
    train_df = pd.read_csv(os.path.join('datasets', f'dataset_{NUMBER_NODES}_train.csv'))
    val_df = pd.read_csv(os.path.join('datasets', f'dataset_{NUMBER_NODES}_val.csv'))
    test_df = pd.read_csv(os.path.join('datasets', f'dataset_{NUMBER_NODES}_test.csv'))

    featuresNumber = (NUMBER_NODES * NUMBER_NODES - NUMBER_NODES) // 2 
    def get_tuple_tensor_dataset(row):
        X = row[0 : featuresNumber].astype('float32')
        Y = row[featuresNumber: ].astype('float32') # Inclui a banda otima na posicao 0
        return torch.from_numpy(X), torch.from_numpy(Y)

    train_dataset = list(map(get_tuple_tensor_dataset, train_df.to_numpy()))
    val_dataset = list(map(get_tuple_tensor_dataset, val_df.to_numpy()))
    test_dataset = list(map(get_tuple_tensor_dataset, test_df.to_numpy()))
    return train_dataset, val_dataset, test_dataset

# %%
train_data, val_data, test_data = load_data()

# %%
BATCH_SIZE = 64
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# %%
for x, y in train_dataloader:
    print("Shape of X [Batches, Digits in each Batch]: ", x.shape)
    print("Grad = ", x.requires_grad)
    print("Shape of y [Batches, Optimal labels in each Batch]: ", y.shape)
    break

# %%
def get_bandwidth_nn_output(Graph, nodelist):
    Graph = np.array(Graph, dtype=np.int32)
    Graph = nx.Graph(Graph)
    L = nx.laplacian_matrix(Graph, nodelist=nodelist.cpu().detach().numpy())
    x, y = np.nonzero(L)
    return (x-y).max()

def getGraph(upperTriangleAdjMatrix):
    dense_adj = np.zeros((NUMBER_NODES, NUMBER_NODES))
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

# %%
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(((NUMBER_NODES * NUMBER_NODES - NUMBER_NODES) // 2 ), 128)
        self.fc2 = nn.Linear(128, NUMBER_NODES)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# %%
class CustomLoss(torch.nn.Module):
    
    def __init__(self):
        super(CustomLoss,self).__init__()

    def loss_repeated_labels(self, roundedOutput):
      # computes the sample variance + (shapeItShouldBe - ShapeItIs)**2
      used_labels, counts = torch.unique(roundedOutput, return_counts=True)
      counts = counts.type(torch.DoubleTensor)
      return torch.var(counts, unbiased=False) + (roundedOutput.shape[0] - counts.shape[0])**2

    def mse_repeated_labels(self, roundedOutput):
      # computes the MSE of ([2., 1., 1.] - [1., 1., 1.])
      # in other words, the error from being an ones_like tensor
      used_labels, counts = torch.unique(roundedOutput, return_counts=True)
      counts = counts.type(torch.DoubleTensor)
      mse_loss = torch.nn.MSELoss()
      return mse_loss(counts, torch.ones_like(counts))

    def levenshtein_distance(self, roundedOutput):
      # computes how many modifications should be done in the tensor in 
      # order to not repeat any label, in any order (just not repeat)
      used_labels, counts = torch.unique(roundedOutput, return_counts=True)
      counts = counts.type(torch.DoubleTensor)
      return torch.sum(counts - 1)

    def forward(self, output, target):
      # computes the sum of:
      # MSE of ([1., 1,. 2.], [0., 1., 2.])
      # sample variance + (shapeItShouldBe - ShapeItIs)**2
      # MSE of ([2., 1., 1.], [1., 1., 1.])
      # how many modifications should be done to avoid repetitions
      labels = np.arange(NUMBER_NODES)
      try:
        roundedOutput = output.round()

      except Exception as e:
        output_band = 2 * target[0]
      loss_mse = ((output - target[1:])**2).mean()

      roundedOutput = output.round()
      loss_repeated = self.loss_repeated_labels(roundedOutput)
      levenshtein = self.levenshtein_distance(roundedOutput)
      mse_ones_like = self.mse_repeated_labels(roundedOutput)
      return loss_mse + loss_repeated + levenshtein + mse_ones_like

# %%
teste = CustomLoss()
y_pred = torch.tensor([0., 1., 1., 2., 2., 3., 1.])
y_true = torch.tensor([0., 0., 1., 2., 3., 4., 5., 6.])
teste.forward(y_pred, y_true)

# %%
def train(dataloader, model, optimizer, epoch):
    criterion = CustomLoss()
    model.train() # turn on possible layers/parts specific for training, like Dropouts for example
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        for input, target in zip(X, y):
            input, target = input.to(device), target.to(device)
            pred = model(input)
            loss = criterion(pred, target)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

    return (train_loss / len(dataloader))

def validate(dataloader, model):
    model.eval() # turn off possible layers/parts specific for training, like Dropouts for example
    eval_loss = 0
    # with torch.no_grad(): # turn off gradients computation
    for x, y in dataloader:
        for input, target in zip(x, y):
            input, target = input.to(device), target.to(device)
            pred = model(input)

            criterion = CustomLoss()
            loss = criterion(pred, target)

            eval_loss += loss.item()
    return (eval_loss / len(dataloader))

# def test(dataloader, model, epoch):
#     num_batches = len(dataloader)
#     model.eval()
#     test_loss, correct = 0, 0
#     with torch.no_grad(): # turn off gradients computation
#         for X, y in dataloader:
#             correct_batch = 0
#             for input, target in zip(X, y):
#                 input, target = input.to(device), target.to(device)
#                 pred = model(input)
#                 criterion = CustomLoss()
#                 loss = criterion(pred, target)
#                 test_loss += loss.item()
#                 t = target[1 : ]
#                 p = pred.round()
#                 correct_batch += t.eq(p).sum().type(torch.float32)
#                 correct_batch /= len(input)
#             correct += correct_batch / len(X)
#     test_loss /= num_batches
#     if epoch % 10 == 0:
#       print(f"Test loss:\nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# %%
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
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
        path = os.path.join(os.path.dirname(__file__), 'checkpoint4.pt')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

# %%
epochs = 10000

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model = NeuralNetwork().to(device)
print(model)

# %%
# TODO: test with lr=0.0003
# REMINDER: already tested with l=0.001; didn't seem to converge
optimizer = optim.SGD(model.parameters(), lr=0.0001)

list_train_loss = []
list_val_loss = []

early_stopping = EarlyStopping(patience=500, verbose=True)

# %%
for epoch in range(epochs):
    # for each epoch, we got a training loss and a validating loss.
    train_loss = train(train_dataloader, model, optimizer, epoch)
    list_train_loss.append(train_loss)
    val_epoch_loss = validate(val_dataloader, model)
    list_val_loss.append(val_epoch_loss)
    print(f'Epoch {epoch + 1}, train_loss: {train_loss}, val_loss: {val_epoch_loss}')

    # early_stopping needs the validation loss to check if it has decresed, 
    # and if it has, it will make a checkpoint of the current model
    valid_loss = np.average(list_val_loss)
    early_stopping(valid_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break
    # test(test_dataloader, model, epoch)

# %%
# torch.save(model.state_dict(), "model.pth")
# print("Saved PyTorch Model State to model.pth")

plt.plot(list_train_loss, label='Training Loss')
plt.plot(list_val_loss, label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc='upper right')
plt.title('Training and Validation Loss average per batch')
plt.savefig(os.path.join(os.path.dirname(__file__), 'loss4.jpg'))
plt.clf()

# %%
for x, y in test_dataloader:
  x, y = x.to(device), y.to(device)
  pred = model(x).round()
  print("=============")
  print('pred:', pred)
  print('y:', y[:,1:])
  break
# load the last checkpoint with the best model
# model.load_state_dict(torch.load('checkpoint.pt'))


