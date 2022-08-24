import math
import os
import argparse
import time

import pandas as pd
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

from ModelInterface import ModelInterface
from Helper import Helper

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

class PointerNetHelper(Helper):
  def __init__(self, MAX_NUMBER_NODES):
    super().__init__(MAX_NUMBER_NODES)

  def logits_to_valid_sequences(self, pred):
    logits_sequences = {}
    true_sequences = {}

    batch_size = pred[0][0].shape[0]

    for i in range(batch_size):
      logits_sequences[i] = []
      true_sequences[i] = []

    for logits_batch, true_batch in pred:
      for batch_id, (logits, true) in enumerate(zip(logits_batch, true_batch)):
        logits_sequences[batch_id].append(logits.cpu())
        true_sequences[batch_id].append(true.cpu())

    seq_len = np.where(np.array(true_sequences[0]) == 35)[0]
    if not len(seq_len):
      seq_len = self.MAX_NUMBER_NODES
    else:
      seq_len = seq_len[0]

    pred_sequences = []
    target_sequences = []
    
    quantity_repeated = 0
    cases_with_repetition = 0
    for batch_id in logits_sequences:
      pred_sequence = []
      isCase_with_repetition = False

      logits_sequences[batch_id] = logits_sequences[batch_id][: seq_len]
      logits_sequences[batch_id] = list(map(lambda x: x.softmax(0)[ : seq_len], logits_sequences[batch_id]))
      
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

    logits_sequences = None

    for batch_id in true_sequences:
      target_sequences.append(true_sequences[batch_id][ : seq_len])

    true_sequences = None

    return pred_sequences, target_sequences, quantity_repeated, cases_with_repetition, seq_len

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

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, MAX_NUMBER_NODES, patience=7, verbose=False, delta=0, ):
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
        self.MAX_NUMBER_NODES = MAX_NUMBER_NODES

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
        path = os.path.join('saved_models', f'PointerNetwork_unified_{self.MAX_NUMBER_NODES}_vertices.pt')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

class Attention(nn.Module):
    # def __init__(self, hidden_size, use_tanh=False, C=10, use_cuda=USE_CUDA):
    def __init__(self, hidden_size,  C, use_cuda, use_tanh=False):
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

class CustomLoss():
  def __init__(self):
    self.mse = nn.MSELoss()

  def loss_repeated_labels(self, sequenceOutput):
    batch_size = sequenceOutput.shape[0]

    used_labels, counts = torch.unique(sequenceOutput, return_counts=True)
    counts = counts.type(torch.FloatTensor)

    counts_shape = counts.shape[0]

    optimalCounts = torch.ones(counts_shape) * batch_size

    return self.mse(counts, optimalCounts)

  def __call__(self, pred, true):
     mse = self.mse(pred, true)
     loss_repeated = self.loss_repeated_labels(pred)
     return mse + loss_repeated

class PointerNet(nn.Module):
    def __init__(self, 
            embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            tanh_exploration,
            use_tanh,
            seq_len_target,
            use_cuda):
        super(PointerNet, self).__init__()
        
        self.embedding_size = embedding_size
        self.hidden_size    = hidden_size
        self.n_glimpses     = n_glimpses
        self.seq_len        = seq_len
        self.use_cuda       = use_cuda

        self.seq_len_target = seq_len_target
        
        self.embedding = nn.Embedding(seq_len, embedding_size)
        self.encoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.pointer = Attention(hidden_size, C=tanh_exploration, use_cuda=use_cuda, use_tanh=use_tanh)
        self.glimpse = Attention(hidden_size, C=tanh_exploration, use_cuda=use_cuda, use_tanh=False)

        self.decoder_start_input = nn.Parameter(torch.FloatTensor(embedding_size))
        self.decoder_start_input.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))
        
        self.criterion = nn.CrossEntropyLoss()
        self.customLoss = CustomLoss()
        
    # def apply_mask_to_logits(self, logits, mask, idxs): 
    #     batch_size = logits.size(0)
    #     clone_mask = mask.clone()

    #     if idxs is not None:
    #         clone_mask[[i for i in range(batch_size)], idxs.data] = 1
    #         logits[clone_mask] = -np.inf
    #     return logits, clone_mask
    def apply_mask_to_logits(self, logits, mask, idxs): 
      clone_mask = mask.clone()
      if idxs is not None:
        for (idx, seq_len) in idxs:
          clone_mask[idx, [range(seq_len, self.seq_len)]] = 1
        logits[clone_mask] = -np.inf
      return logits, clone_mask
    
    def list_of_tuple_with_logits_true_to_verticalSequence(self, item_tuple):
        sequence = []
        softmax = nn.Softmax(dim=1)

        logits = softmax(item_tuple[0])
        true = item_tuple[1].cpu().numpy()

        argmax_indices = torch.argmax(logits, dim=1)
        logits = None
        for i in argmax_indices.cpu().numpy():
            sequence.append(i)

        sequence = np.array(sequence)
        return sequence, true

    def verticalSequence_to_horizontalSequence(self, verticalSequence):
        horizontalSquence = torch.tensor(verticalSequence, dtype=torch.float32)
        return horizontalSquence.permute(2, 1, 0)

    def verticalSequence_to_horizontalSequence_splitted(self, verticalSequence):
        horizontalSquence = torch.tensor(verticalSequence, dtype=torch.float32)
        permuted = horizontalSquence.permute(2, 1, 0)
        pred, true = torch.tensor_split(permuted, 2, dim=1)

        horizontalSquence = None
        permuted = None

        pred = torch.squeeze(pred)
        true = torch.squeeze(true)
        return pred, true
    
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

        mask = torch.zeros(batch_size, seq_len).bool()
        if self.use_cuda:
            mask = mask.cuda()
        """
        masks = (inputs == 2).nonzero()
        idxs = {}
        for (idx, seq_length) in masks:
          idx = int(idx.cpu())
          if idx not in idxs:
            idxs[idx] = int(seq_length.cpu())
        idxs = idxs.items()
        if not len(idxs):
          idxs = None
        """
        idxs = None

        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)
        
        loss = 0
        
        output = []
        for i in range(self.seq_len_target):
            _, (hidden, context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))
            
            query = hidden.squeeze(0)
            for _ in range(self.n_glimpses):
                ref, logits = self.glimpse(query, encoder_outputs)
                logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
                query = torch.bmm(ref, F.softmax(logits, dim=1).unsqueeze(2)).squeeze(2)    
                
            _, logits = self.pointer(query, encoder_outputs)
            logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
            
            decoder_input = target_embedded[:,i,:]

            output.append((logits, target[ : , i]))

            """
            if idxs != None, apply_mask_to_logits will run, and put -np.inf on invalid logits.

            the problem with that is when it is executed self.criterion(logits, target[:,i]),
            target[:,i] on an invalid node will evaluate to 35 (the chosen mask), but the
            probability of being 35 will be -np.inf, causing the loss to be infinity.

            I thought that before calculating the loss I could split the logits to its
            valid seq_len, it would cause logits[i] to have maximum seq_len as len(logits[i]).

            But logits are applicable for a batch of graphs, imagining that logits[0] are logits
            for a label of a 9 vertices graph and logits[1] are logits for a label of a 7 vertices
            graph, we could not have logits with shape (batch_size, 7 or 9) at the same time.

            that's why I set up idxs = None, in that way the PNet will treat all logits as valids
            and try to approximate logits to the 35 label (the chosen mask). The logits of a 
            prediction and target are treat in the logits_to_valid_sequences helper function.
            Since it is handling just a a single graph, instead of a batch of graphs, it can
            make the split in the logits and the target
            """

            loss += self.criterion(logits, target[:,i])
            
        loss_output =  loss / self.seq_len_target

        verticalSequences = np.array(list(map(self.list_of_tuple_with_logits_true_to_verticalSequence, output)))
        pred_sequences, true_sequences = self.verticalSequence_to_horizontalSequence_splitted(verticalSequences)
        custom_loss = self.customLoss(pred_sequences, true_sequences)

        return output, loss_output + custom_loss

class PointerNetwork(ModelInterface):
  def __init__(self, MAX_NUMBER_NODES, batch_size, epochs):
    self.MAX_NUMBER_NODES = MAX_NUMBER_NODES
    self.features_length = (self.MAX_NUMBER_NODES * self.MAX_NUMBER_NODES - self.MAX_NUMBER_NODES) // 2
    self.batch_size = batch_size
    self.epochs = epochs

  def load_train_data(self):
    train_df = pd.read_csv(os.path.join('..', 'Datasets', f'dataset_7_9_train(2and35_as_mask).csv'))
    val_df = pd.read_csv(os.path.join('..', 'Datasets', f'dataset_7_9_val(2and35_as_mask).csv'))

    train_df = pd.concat((train_df, val_df))

    def get_tuple_tensor_dataset(row):
        X = row[0 : self.features_length].astype('int32')
        Y = row[self.features_length + 1: ].astype('int32') # Pula a banda otima na posicao 0
        X = torch.from_numpy(X).type(torch.long)
        Y = torch.from_numpy(Y).type(torch.long)
        return X, Y

    train_dataset = list(map(get_tuple_tensor_dataset, train_df.to_numpy()))
    return train_dataset

  def load_test_data(self):
    test_df = pd.read_csv(os.path.join('..', 'Datasets', f'dataset_5_7_9_test(2and35_as_mask).csv'))

    def get_tuple_tensor_dataset(row):
        X = row[0 : self.features_length].astype('int32')
        Y = row[self.features_length + 1: ].astype('int32') # Pula a banda otima na posicao 0
        X = torch.from_numpy(X).type(torch.long)
        Y = torch.from_numpy(Y).type(torch.long)
        return X, Y

    test_dataset = list(map(get_tuple_tensor_dataset, test_df.to_numpy()))
    return test_dataset

  def train(self, train_loader, model, optimizer):
    train_loss = 0
    model.train()
    for x, y in train_loader:
      optimizer.zero_grad()

      logits_with_target_of_a_sequence, loss_output = model(x, y)
      loss_output.backward()

      train_loss += loss_output.item()

      optimizer.step()

    return train_loss

  def validate(self, val_loader, model):
    eval_loss = 0
    model.eval()
    with torch.no_grad():
      for x, y in val_loader:
          logits_with_target_of_a_sequence, loss_output = model(x, y)
          eval_loss += loss_output.item()
    return eval_loss

  def get_predicts(self, test_dataloader, model):
    preds = []
    model.eval()
    with torch.no_grad():
      for x, y in test_dataloader:
        logits_with_target_of_a_sequence, loss_output = model(x, y)
        preds.append((x, logits_with_target_of_a_sequence))
    return preds

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
    # helps garbage collection
    
    train_dataloader = DeviceDataLoader(train_dataloader, self.device)
    test_dataloader = DeviceDataLoader(test_dataloader, self.device)

    model = PointerNet(
      embedding_size=32,
      hidden_size=32,
      seq_len=self.features_length,
      n_glimpses=1,
      tanh_exploration=self.features_length,
      use_tanh=True,
      seq_len_target=self.MAX_NUMBER_NODES,
      use_cuda=True if str(self.device) == 'cuda' else False
    ).to(self.device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    early_stopping = EarlyStopping(MAX_NUMBER_NODES=self.MAX_NUMBER_NODES, patience=500, verbose=True)

    if not os.path.exists('saved_models'):
      os.makedirs('saved_models')

    train_loss = []
    val_loss   = []

    for epoch in range(self.epochs):
      epoch_train_loss = self.train(train_dataloader, model, optimizer)
      epoch_val_loss = self.validate(test_dataloader, model)
      
      train_loss.append(epoch_train_loss)
      val_loss.append(epoch_val_loss)
      print(f'Epoch {epoch + 1}, train_loss: {epoch_train_loss}, val_loss: {epoch_val_loss}')

      valid_loss = np.average(val_loss)
      early_stopping(valid_loss, model)
      if early_stopping.early_stop:
          print("Early stopping")
          break

    train_dataloader = None
    test_dataloader = None

    if not os.path.exists('plotted_figures'):
      os.makedirs('plotted_figures')

    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join('plotted_figures', f'PointerNetwork_loss_unified_{self.MAX_NUMBER_NODES}_vertices.jpg'))
    plt.clf()

  def predict(self):
    try:
      helper = PointerNetHelper(self.MAX_NUMBER_NODES)
      self.device = get_default_device()
      print(f"Using {self.device} device")

      model = PointerNet(
        embedding_size=32,
        hidden_size=32,
        seq_len=self.features_length,
        n_glimpses=1,
        tanh_exploration=self.features_length,
        use_tanh=True,
        seq_len_target=self.MAX_NUMBER_NODES,
        use_cuda=True if self.device == 'cuda' else False
      ).to(self.device)

      path = os.path.join('saved_models', f'PointerNetwork_unified_{self.MAX_NUMBER_NODES}_vertices.pt')
      model.load_state_dict(torch.load(path))

      BATCH_SIZE =  self.batch_size
      test_data = self.load_test_data()
      test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
      
      test_data = None

      test_dataloader = DeviceDataLoader(test_dataloader, self.device)

      # preds = self.get_predicts(test_dataloader, model)

      original_bandwidths = {
        5: [],
        7: [],
        9: []
      }

      pred_bandwidths = {
        5: [],
        7: [],
        9: []
      }

      target_bandwidths = {
        5: [],
        7: [],
        9: []
      }

      prediction_times = []

      count = 0
      cases_with_repetition = 0

      model.eval()
      with torch.no_grad():
        for input_data, target_data in test_dataloader:
          for x, y in zip(input_data, target_data):
            start_time = time.time()

            # if 2 not in x:
            #   continue
            # if 2 in x:
            #   continue

            prediction, _ = model(x.unsqueeze(dim=0), y.unsqueeze(dim=0))

            preds, targets, q, c, seq_len = helper.logits_to_valid_sequences(prediction)
            
            count += q
            cases_with_repetition += c

            prediction_times.append(time.time() - start_time)

            x = x.cpu()

            graph = helper.getGraph(x)
            graph = nx.Graph(graph)

            original_band = helper.get_bandwidth(graph, np.array(None))
            original_bandwidths[seq_len].append(original_band)

            pred_band = helper.get_bandwidth(graph, np.array(preds[0]))
            pred_bandwidths[seq_len].append(pred_band)

            targets = torch.tensor(targets).cpu()

            true_band = helper.get_bandwidth(graph, np.array(targets[0]))
            target_bandwidths[seq_len].append(true_band)

      test_length = 0
      for x, y in test_dataloader:
        test_length += x.shape[0]
      test_dataloader = None
      print(test_length)

      PointerNetworkResult = helper.getResult(
        model_name='PointerNetwork',
        original_bandwidths=original_bandwidths,
        pred_bandwidths=pred_bandwidths,
        target_bandwidths=target_bandwidths,
        count=count,
        cases_with_repetition=cases_with_repetition,
        prediction_times=prediction_times
      )
      
      return PointerNetworkResult

    except FileNotFoundError as e:
      print(e)
      print('Error loading the model from disk, should run model.fit')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Pytorch - Pointer Network')
  parser.add_argument('-mv','--max_vertices', help='maximum number of vertices dataset [9]', required=True)
  parser.add_argument('-m','--mode', help='0 - fit, 1 - predict', required=True)
  parser.add_argument('-b','--batch', help="batch_size - 32, 64, 128, 256, ...", required=True)
  parser.add_argument('-e','--epochs', help="epochs - 52, 64, 128, ...", required=True)
  args = parser.parse_args()

  MAX_NUMBER_NODES = int(args.max_vertices)
  batch_size = int(args.batch)
  epochs = int(args.epochs)

  pointerNetwork = PointerNetwork(MAX_NUMBER_NODES=MAX_NUMBER_NODES, batch_size=batch_size, epochs=epochs)

  if args.mode == '0':
    pointerNetwork.fit()
  if args.mode == '1':
    df_result, df_metadata = pointerNetwork.predict()
    print(df_result.to_latex())
    print(df_metadata.to_latex())