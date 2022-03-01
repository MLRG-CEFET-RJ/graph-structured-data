# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
import random
import argparse
import networkx as nx
import os

# %%
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

parser = argparse.ArgumentParser(description='Pytorch - Deep learning custom loss')
parser.add_argument('-v','--vertices', help='number of vertices dataset [7, 9]', required=True)
parser.add_argument('-n','--name', help='run custom name [ex.: run_5nodes_v1]', required=True)
args = parser.parse_args()

# %%
NUMBER_NODES = int(args.vertices)

# %%
def load_data():
    train_df = pd.read_csv(os.path.join('..', 'datasets', f'dataset_{NUMBER_NODES}_train.csv'))
    val_df = pd.read_csv(os.path.join('..', 'datasets', f'dataset_{NUMBER_NODES}_val.csv'))
    test_df = pd.read_csv(os.path.join('..', 'datasets', f'dataset_{NUMBER_NODES}_test.csv'))

    featuresNumber = (NUMBER_NODES * NUMBER_NODES - NUMBER_NODES) // 2 
    def get_tuple_tensor_dataset(row):
        X = row[0 : featuresNumber].astype('float32')
        Y = row[featuresNumber: ].astype('float32') # Inclui a banda otima na posicao 0
        return torch.from_numpy(X), torch.from_numpy(Y)

    train_df = pd.concat((train_df, val_df))

    train_dataset = list(map(get_tuple_tensor_dataset, train_df.to_numpy()))
    test_dataset = list(map(get_tuple_tensor_dataset, test_df.to_numpy()))
    return train_dataset, test_dataset

# %%
train_data, test_data = load_data()

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

# %%
BATCH_SIZE = 64
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

device = get_default_device()
print(f"Using {device} device")

train_dataloader = DeviceDataLoader(train_dataloader, device)
test_dataloader = DeviceDataLoader(test_dataloader, device)

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
      return loss_mse + loss_repeated + levenshtein

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
        path = os.path.join(os.path.dirname(__file__), f'checkpoint5_{args.vertices}nodes_{args.name}.pt')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

# %%
epochs = 20000

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
    val_epoch_loss = validate(test_dataloader, model)
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
plt.savefig(os.path.join(os.path.dirname(__file__), f'loss5_{args.vertices}nodes_{args.name}.jpg'))
plt.clf()

def count_repeats(output):
    counts = np.unique(np.round(output))
    repeated = NUMBER_NODES - counts.shape[0]
    return repeated

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

def get_bandwidth(Graph, nodelist=None):
    Graph = nx.Graph(Graph)
    if not Graph.edges:
        return 0
    if nodelist.all() != None:
        L = nx.laplacian_matrix(Graph, nodelist=nodelist)
    else:
        L = nx.laplacian_matrix(Graph)
    x, y = np.nonzero(L)
    return (x-y).max()

def get_valid_pred(pred):
    valid = np.ones(NUMBER_NODES)
    labels = np.arange(0, NUMBER_NODES)
    for i in labels:
        min_value = np.amin(pred)
        min_idx, = np.where(pred == min_value)
        min_idx = min_idx[0]
        pred[min_idx] = 100
        valid[min_idx] = i
    return valid

sumTest_original = 0
sumTest_pred = 0
sumTest_true = 0

count = 0
at_least_one_repetition = 0

model = NeuralNetwork().to(device)
path = os.path.join(os.path.dirname(__file__), f'checkpoint5_{args.vertices}nodes_{args.name}.pt')
model.load_state_dict(torch.load(path))

for x, y in test_dataloader:
  x, y = x.to(device), y.to(device)
  output = model(x)

  x = x.cpu()
  output = output.cpu()
  y = y.cpu()

  for features, pred, target in zip(x, output, y):
    features = features.detach().numpy()
    pred = pred.detach().numpy()
    target = target.detach().numpy()[1:]

    print('pred:', np.round(pred))
    print('y:', target)
    repeated_amount = count_repeats(pred)

    if repeated_amount != 0:
      at_least_one_repetition += 1
    pred = get_valid_pred(pred)
    print('pred valid:', pred)
    count += repeated_amount

    graph = getGraph(features)
    original_band = get_bandwidth(graph, np.array(None))
    sumTest_original += original_band
    pred_band = get_bandwidth(graph, pred)
    sumTest_pred += pred_band
    true_band = get_bandwidth(graph, target)
    sumTest_true += true_band
    print("Bandwidth")
    print(original_band)
    print(pred_band)
    print(true_band)

test_length = 0
for x, y in test_dataloader:
    test_length += y.shape[0]

print('Quantidade de rótulos repetidos, exemplo [1, 1, 1, 1, 1, 1, 1] conta como 6 - ', count)
print('Quantidade de saídas com repetição, exemplo [1, 1, 1, 1, 1, 1, 1] conta como 1 - ', at_least_one_repetition)
print("Test length - ", test_length)
print("Bandwidth mean")
print(sumTest_original / test_length)
print("Pred bandwidth mean")
print(sumTest_pred / test_length)
print("True bandwidth mean")
print(sumTest_true / test_length)
