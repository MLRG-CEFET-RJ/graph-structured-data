# %% [markdown]
# This tutorial demostrates Pointer Networks with readable code.

# %%
from lib2to3.pgen2.token import NUMBER
import math
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import os
import networkx as nx
import argparse

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Pytorch - Deep learning custom loss')
parser.add_argument('-v','--vertices', help='number of vertices dataset [7, 9]', required=True)
parser.add_argument('-n','--name', help='run custom name [ex.: run_5nodes_v1]', required=True)
parser.add_argument('-b','--batch', help='batch size', required=True)
parser.add_argument('-c','--cuda', help='[True, False]', required=True)
parser.add_argument('-e','--epochs', help='e.g. [64, 128, ...]', required=True)
args = parser.parse_args()

USE_CUDA = eval(args.cuda)
# %%
BATCH_SIZE = int(args.batch)
NUMBER_NODES = int(args.vertices)
FEATURES_NUMBER_ADJ = (NUMBER_NODES * NUMBER_NODES - NUMBER_NODES) // 2 
FEATURES_NUMBER = FEATURES_NUMBER_ADJ + NUMBER_NODES

def getDegree(graph):
    G = nx.Graph(graph)
    degree = list(dict(G.degree()).values())
    return np.array(degree)

def getGraph(upperTriangleAdjMatrix):
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

def load_data():
    train_df = pd.read_csv(os.path.join('..', '..', 'DNN', 'datasets', f'dataset_{NUMBER_NODES}_train.csv'))
    val_df = pd.read_csv(os.path.join('..', '..', 'DNN', 'datasets', f'dataset_{NUMBER_NODES}_val.csv'))
    test_df = pd.read_csv(os.path.join('..', '..', 'DNN', 'datasets', f'dataset_{NUMBER_NODES}_test.csv'))

    def get_tuple_tensor_dataset(row):
        X = row[0 : FEATURES_NUMBER_ADJ].astype('int32')
        Y = row[FEATURES_NUMBER_ADJ + 1 : ].astype('int32') # FEATURES_NUMBER + 1 Skips the optimal_band value

        X = torch.from_numpy(X)
        upperTriangle = X.type(torch.long)

        graph = getGraph(X)
        degree = getDegree(graph)
        degree = torch.from_numpy(degree)
        degree = degree.type(torch.long)
        X = torch.cat((upperTriangle, degree))

        Y = torch.from_numpy(Y)
        Y = Y.type(torch.long)
        return X, Y

    train_df = pd.concat((train_df, val_df))

    train_dataset = list(map(get_tuple_tensor_dataset, train_df.to_numpy()))
    test_dataset = list(map(get_tuple_tensor_dataset, test_df.to_numpy()))
    return train_dataset, test_dataset

train_data, test_data = load_data()

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

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

device = get_default_device()
print(f"Using {device} device")
print(f"USE_CUDA - {USE_CUDA}")

train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)

for x, y in train_loader:
    print(x[ : , FEATURES_NUMBER_ADJ : ].shape)
    print(x.shape)
    print(y.shape)
    break

seq_len = FEATURES_NUMBER
embedding_size = 3
hidden_size = 3

batch_size = 32

n_glimpses = 1
tanh_exploration = 10

class Attention(nn.Module):
    # def __init__(self, hidden_size, use_tanh=False, C=10, use_cuda=USE_CUDA):
    def __init__(self, hidden_size, use_tanh=False, C=NUMBER_NODES, use_cuda=USE_CUDA):
        super(Attention, self).__init__()
        
        self.use_tanh = use_tanh
        self.W_query = nn.Linear(hidden_size, hidden_size)
        self.W_ref   = nn.Conv1d(hidden_size, hidden_size, 1, 1)
        self.C = C
        
        V = torch.FloatTensor(hidden_size)
        if use_cuda:
            V = V.cuda()  
        self.V = nn.Parameter(V)
        self.V.data.uniform_(-(1. / math.sqrt(hidden_size)) , 1. / math.sqrt(hidden_size))
        
    def forward(self, query, ref):
        """
        Args: 
            query: [batch_size x hidden_size]
            ref:   ]batch_size x seq_len x hidden_size]
        """
        
        batch_size = ref.size(0)
        seq_len    = ref.size(1)

        ref = ref.permute(0, 2, 1)
        query = self.W_query(query).unsqueeze(2)  # [batch_size x hidden_size x 1]
        ref   = self.W_ref(ref)  # [batch_size x hidden_size x seq_len]

        expanded_query = query.repeat(1, 1, seq_len) # [batch_size x hidden_size x seq_len]
        V = self.V.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size x 1 x hidden_size]

        logits = torch.bmm(V, torch.tanh(expanded_query + ref)).squeeze(1)
        
        if self.use_tanh:
            logits = self.C * torch.tanh(logits)
        else:
            logits = logits  
        return ref, logits

class PointerNetLossOutside(nn.Module):
    def __init__(self, 
            embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            tanh_exploration,
            use_tanh,
            seq_len_target,
            use_cuda=USE_CUDA):
        super(PointerNetLossOutside, self).__init__()
        
        self.embedding_size = embedding_size
        self.hidden_size    = hidden_size
        self.n_glimpses     = n_glimpses
        self.seq_len        = seq_len
        self.use_cuda       = use_cuda

        self.seq_len_target = seq_len_target
        
        self.embedding = nn.Embedding(seq_len, embedding_size)
        self.encoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.pointer = Attention(hidden_size, use_tanh=use_tanh, C=tanh_exploration, use_cuda=use_cuda)
        self.glimpse = Attention(hidden_size, use_tanh=False, use_cuda=use_cuda)
        
        self.decoder_start_input = nn.Parameter(torch.FloatTensor(embedding_size))
        self.decoder_start_input.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))
        
        self.criterion = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        
    def apply_mask_to_logits(self, logits, mask, idxs): 
        batch_size = logits.size(0)
        clone_mask = mask.clone()

        if idxs is not None:
            clone_mask[[i for i in range(batch_size)], idxs.data] = 1
            logits[clone_mask] = -np.inf
        return logits, clone_mask
    
    def list_of_tuple_with_logits_true_to_verticalSequence(self, item_tuple):
        sequence = []
        softmax = nn.Softmax(dim=1)

        logits = softmax(item_tuple[0])
        true = item_tuple[1]

        argmax_indices = torch.argmax(softmax(logits), dim=1)
        for i in argmax_indices:
            sequence.append(i)

        sequence = torch.tensor(sequence)
        return sequence, true

    def verticalSequence_to_horizontalSequence(self, verticalSequence):
        pred_batch = []
        true_batch = []
        for stackedPred, stackedTrue in verticalSequence:
            if isinstance(stackedPred, torch.Tensor):
              stackedPred = stackedPred.cpu()
            if isinstance(stackedTrue, torch.Tensor):
              stackedTrue = stackedTrue.cpu()
            pred_batch.append(list(stackedPred.numpy()))
            true_batch.append(list(stackedTrue.numpy()))

        pred_batch = torch.tensor(pred_batch)
        pred_batch = pred_batch.permute(1, 0)

        true_batch = torch.tensor(true_batch)
        true_batch = true_batch.permute(1, 0)

        data = []

        for pred, true in zip(pred_batch, true_batch):
            data.append((pred, true))
        return data

    def verticalSequence_to_horizontalSequence_splitted(self, verticalSequence):
        pred_batch = []
        true_batch = []
        for stackedPred, stackedTrue in verticalSequence:
            if isinstance(stackedPred, torch.Tensor):
              stackedPred = stackedPred.cpu()
            if isinstance(stackedTrue, torch.Tensor):
              stackedTrue = stackedTrue.cpu()
            pred_batch.append(list(stackedPred.numpy()))
            true_batch.append(list(stackedTrue.numpy()))

        pred_batch = torch.tensor(pred_batch)
        pred_batch = pred_batch.permute(1, 0)

        true_batch = torch.tensor(true_batch)
        true_batch = true_batch.permute(1, 0)

        pred_batch = pred_batch.type(torch.FloatTensor)
        true_batch = true_batch.type(torch.FloatTensor)

        return pred_batch, true_batch

    def loss_repeated_labels(self, sequenceOutput):
      batch_size = sequenceOutput.shape[0]

      used_labels, counts = torch.unique(sequenceOutput, return_counts=True)
      counts = counts.type(torch.FloatTensor)

      counts_shape = counts.shape[0]
      # output_shape = roundedOutput.shape[1]

      optimalCounts = torch.ones(counts_shape) * batch_size

      # return ((counts - optimalCounts)**2).mean() + (output_shape - counts_shape)
      # return torch.var(counts, unbiased=False)
      return self.mse(counts, optimalCounts)
    
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

    def forward(self, inputs, target):
        """
        Args: 
            inputs: [batch_size x sourceL]
        """
        batch_size = inputs.size(0)
        seq_len    = inputs.size(1)
        assert seq_len == self.seq_len
        
        embedded = self.embedding(inputs)
        target_embedded = self.embedding(target)
        encoder_outputs, (hidden, context) = self.encoder(embedded)
        
        mask = torch.zeros(batch_size, seq_len).byte()
        if self.use_cuda:
            mask = mask.cuda()
            
        idxs = None
       
        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)
        
        loss = 0
        
        output = []
        # for i in range(seq_len):
        for i in range(self.seq_len_target):
            
            _, (hidden, context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))
            
            query = hidden.squeeze(0)
            for _ in range(self.n_glimpses):
                ref, logits = self.glimpse(query, encoder_outputs)
                logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
                # even without the line above, the model make 5 zeros for the last 5 logits
                # query = torch.bmm(ref, F.softmax(logits).unsqueeze(2)).squeeze(2) 
                query = torch.bmm(ref, logits.softmax(dim=0).unsqueeze(2)).squeeze(2) 
                
                
            _, logits = self.pointer(query, encoder_outputs)
            logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
            # even without the line above, the model make 5 zeros for the last 5 logits
            
            decoder_input = target_embedded[:,i,:]

            output.append((logits, target[ : , i]))

            loss += self.criterion(logits, target[:,i])
            
        loss_output =  loss / self.seq_len_target

        verticalSequences = list(map(self.list_of_tuple_with_logits_true_to_verticalSequence, output))
        pred_sequences, true_sequences = self.verticalSequence_to_horizontalSequence_splitted(verticalSequences)

        mse = self.mse(pred_sequences, true_sequences)
        loss_repeated = self.loss_repeated_labels(pred_sequences)
        custom_loss = mse + loss_repeated

        return output, loss_output + custom_loss

# %%
def train(train_loader, model, optimizer):
  loss = 0
  model.train()
  for batch, (x, y) in enumerate(train_loader):
    optimizer.zero_grad()

    x = x[ : , FEATURES_NUMBER_ADJ : ]
    logits_with_target_of_a_sequence, loss_output = model(x, y)
    loss_output.backward()

    loss += loss_output.item()

    optimizer.step()

    if batch % 100 == 0:
      print(f"Loss: {loss}, batch: {batch} ")
  return loss

def validate(val_loader, model):
  loss = 0
  model.eval()
  for batch, (x, y) in enumerate(val_loader):
    x = x[ : , FEATURES_NUMBER_ADJ : ]
    logits_with_target_of_a_sequence, loss_output = model(x, y)

    loss += loss_output.item()
  return loss
  
def predict(val_loader, model):
  preds = []
  model.eval()
  for batch, (x, y) in enumerate(val_loader):
    forward_x = x[ : , FEATURES_NUMBER_ADJ : ]
    logits_with_target_of_a_sequence, loss_output = model(forward_x, y)

    test_x = x[ : , : FEATURES_NUMBER_ADJ]
    preds.append((test_x, logits_with_target_of_a_sequence))
  return preds
  # https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
  # https://www.tensorflow.org/tutorials/images/classification?authuser=1#download_and_explore_the_dataset 
  # the link above is without softmax in the model, but has softmax when prediciting
  # https://www.tensorflow.org/tutorials/keras/classification
  # the link above is with softmax in the model, thus has no softmax when prediciting

# %%
n_epochs = int(args.epochs)
train_loss = []
val_loss   = []

pointer_modified = PointerNetLossOutside(
    embedding_size=32,
    hidden_size=32,
    seq_len=NUMBER_NODES,
    n_glimpses=1,
    tanh_exploration=tanh_exploration,
    use_tanh=True,
    seq_len_target=NUMBER_NODES
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(pointer_modified.parameters(), lr=1e-4)

for epoch in range(n_epochs):
    print(f"epoch: {epoch + 1}")
    epoch_train_loss = train(train_loader, pointer_modified, optimizer)
    epoch_val_loss = validate(val_loader, pointer_modified)
    
    train_loss.append(epoch_train_loss)
    val_loss.append(epoch_val_loss)

# %%
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc='upper right')
plt.title('Loss')
plt.savefig(f'pointernetwork_degree_loss_{args.name}.jpg')
plt.clf()

# %%
preds = predict(val_loader, pointer_modified)

softmax = nn.Softmax(dim=1)

it = iter(preds)
input_data, pred = next(it)

# %%
input_data

# %%
pred

# %%
# this function is in PointerNetLossOutside, repeated on purpose for visualize the returned data
def list_of_tuple_with_logits_true_to_verticalSequence(item_tuple):
  sequence = []

  logits = softmax(item_tuple[0])
  true = item_tuple[1]

  argmax_indices = torch.argmax(softmax(logits), dim=1)
  for i in argmax_indices:
    sequence.append(i)

  sequence = torch.tensor(sequence)
  return sequence, true

# this function is also in PointerNetLossOutside, repeated on purpose for visualize the returned data
def verticalSequence_to_horizontalSequence(verticalSequence):
  pred_batch = []
  true_batch = []
  for stackedPred, stackedTrue in verticalSequence:
    if isinstance(stackedPred, torch.Tensor):
      stackedPred = stackedPred.cpu()
    if isinstance(stackedTrue, torch.Tensor):
      stackedTrue = stackedTrue.cpu()
    pred_batch.append(list(stackedPred.numpy()))
    true_batch.append(list(stackedTrue.numpy()))

  pred_batch = torch.tensor(pred_batch)
  pred_batch = pred_batch.permute(1, 0)

  true_batch = torch.tensor(true_batch)
  true_batch = true_batch.permute(1, 0)

  data = []

  for pred, true in zip(pred_batch, true_batch):
    data.append((pred, true))
  return data

# [4, 3, 6, 5, 5, 5, 2]
def list_of_tuple_with_logits_true_to_sequences(pred):
  logits_sequences = {}
  true_sequences = {}

  batch_size = pred[0][0].shape[0]

  for i in range(batch_size):
    logits_sequences[str(i)] = []
    true_sequences[str(i)] = []

  for logits_batch, true_batch in pred:
    for batch_id, (logits, true) in enumerate(zip(logits_batch, true_batch)):
      logits_sequences[str(batch_id)].append(logits)
      true_sequences[str(batch_id)].append(true)

  pred_sequences = []
  target_sequences = []
  
  quantity_repeated = 0
  cases_with_repetition = 0
  for batch_id in logits_sequences:
    pred_sequence = []
    isCase_with_repetition = False

    logits_sequences[batch_id] = list(map(lambda x: x.softmax(0), logits_sequences[batch_id]))
    for logits in logits_sequences[batch_id]:
      appended = False
      while(not appended):
        argmax_indice = torch.argmax(logits, dim=0)
        if argmax_indice in pred_sequence:
          logits[argmax_indice] = -1 # argmax already used = -1 (softmax is [0, 1])
          quantity_repeated += 1
          if not isCase_with_repetition:
            cases_with_repetition += 1
            isCase_with_repetition = True
        else:
          pred_sequence.append(argmax_indice)
          appended = True
    pred_sequences.append(pred_sequence)

  for batch_id in true_sequences:
    target_sequences.append(true_sequences[batch_id])
  return pred_sequences, target_sequences, quantity_repeated, cases_with_repetition


# %%
verticalSequences = list(map(list_of_tuple_with_logits_true_to_verticalSequence, pred))
verticalSequences
horizontalSequences = verticalSequence_to_horizontalSequence(verticalSequences)
horizontalSequences

# %%
pred_sequences, target_sequences, quantity_repeated, cases_with_repetition = list_of_tuple_with_logits_true_to_sequences(pred)

def getGraph(upperTriangleAdjMatrix):
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

def get_bandwidth(Graph, nodelist):
    Graph = nx.Graph(Graph)
    if not Graph.edges:
        return 0
    if nodelist.all() != None:
        L = nx.laplacian_matrix(Graph, nodelist=nodelist)
    else:
        L = nx.laplacian_matrix(Graph)
    x, y = np.nonzero(L)
    return (x-y).max()

def get_valid_sequence(output):
  maximum = FEATURES_NUMBER - 1
  maximum_valid = NUMBER_NODES - 1

  valid_output = np.ones(NUMBER_NODES)
  for _ in range(NUMBER_NODES):
    while(maximum not in output):
      maximum -= 1
    index = output.index(maximum)
    output[index] = FEATURES_NUMBER
    valid_output[index] = maximum_valid
    maximum_valid -= 1
  
  return valid_output

"""
    the list_of_tuple_with_logits_true_to_sequences algorithm ensures that the sequence will be different numbers
    but does not ensures that could not get a output like that:
    [0, 2, 1, 3, 5, 4, 7] # the correct range is [0, 6]
    fix that
"""
print(get_valid_sequence([0, 2, 1, 3, 5, 4, 7]))
print(get_valid_sequence([0, 1, 2, 3, 4, 5, 10]))

# %%
preds = predict(val_loader, pointer_modified)

sumTest_original = []
sumTest_pred = []
sumTest_true = []

count_total = 0
cases_with_repetition_total = 0

start = time.time()
for input_data, pred in preds:
  pred_sequences, target_sequences, quantity_repeated, cases_with_repetition = list_of_tuple_with_logits_true_to_sequences(pred)
  for x, output, true in zip(input_data, pred_sequences, target_sequences):
    """
    print(x)
    print(output)
    print(true)
    """

    count_total += quantity_repeated
    cases_with_repetition_total += cases_with_repetition

    output = get_valid_sequence(output)

    graph = getGraph(x)
    original_band = get_bandwidth(graph, np.array(None))
    sumTest_original.append(original_band)

    pred_band = get_bandwidth(graph, np.array(output))
    sumTest_pred.append(pred_band)

    true = torch.tensor(true).cpu()

    true_band = get_bandwidth(graph, np.array(true))
    sumTest_true.append(true_band)

    # print("Bandwidth")
    # print(original_band)
    # print(pred_band)
    # print(true_band)
end = time.time()

# %%
total_length = 0
for i in preds:
  total_length += i[0].shape[0]

# %%
print('Quantidade de rótulos repetidos, exemplo [1, 1, 1, 1, 1, 1, 1] conta como 6 - ', count_total)
print('Quantidade de saídas com repetição, exemplo [1, 1, 1, 1, 1, 1, 1] conta como 1 - ', cases_with_repetition)
test_length = total_length

print('Test length - ', test_length)
print('Tempo medio - ', (end - start) / test_length)
print("Bandwidth mean±std")
print(f'{np.mean(sumTest_original)}±{np.std(sumTest_original)}')
print("Pred bandwidth mean±std")
print(f'{np.mean(sumTest_pred)}±{np.std(sumTest_pred)}')
print("True bandwidth mean±std")
print(f'{np.mean(sumTest_true)}±{np.std(sumTest_true)}')


