import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator

from sklearn import preprocessing

import timeit

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())

def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    # armazena a classe de cada vertice -> shape(2708,1)
    # o valor em cada posicao referencia label_map
    labels = np.empty((num_nodes,1), dtype=np.int64)
    # mapa onde chave referencia o id original do vertice e value referencia o indice em feat_data
    # por exemplo, node_map['31336'] = 0 (ou seja, feat_data[0] possui as features do vertice de id 31336
    node_map = {}
    # mapa onde chave referencia o nome original da classe e value eh referenciado em labels
    label_map = {}
    with open("cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = map(float, info[1:-1])
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
            
    return feat_data, labels, adj_lists

def run_cora(printout = True):
    
    begin = timeit.default_timer()
    
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708
    feat_data, labels, adj_lists = load_cora()
    
    ################################################################################################################
    # sets definition earlier
    rand_indices = np.random.permutation(num_nodes) # len(rand_indices) = 2708
    test = rand_indices[:1000]        # 1000 examples
    val = rand_indices[1000:1500]     # 1500 examples
    # train = list(rand_indices[1500:]) # 1208 examples              # ALTERADO PARA USAR SOMENTE 10% DOS EXEMPLOS DE TREINO
    train = list(rand_indices[1500:1620]) # 120 examples
    
    ################################################################################################################
    # normalization
    ################################################################################################################
    scaler = preprocessing.StandardScaler().fit(feat_data[train]) # only fit in the train examples
    feat_data = scaler.transform(feat_data)
    ################################################################################################################
    ################################################################################################################
    
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    # MeanAggregator params
    #  features, cuda=False, gcn=False

    # Encoder params
    #  features, feature_dim, embed_dim, adj_lists, aggregator, num_sample=10, base_model=None, 
    #  gcn=False, cuda=False, feature_transform=False
    
    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(7, enc2)
#    graphsage.cuda()
    
    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    
    ##################################################################################################
    train_loss = list()   # inicializa o vetor
    
    for batch in range(100):
        #batch_nodes = train[:256]                                   # ALTERADO PARA USAR SOMENTE 10% DOS EXEMPLOS DE TREINO
        batch_nodes = train[:25]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        ##################################################################################################
        train_loss.append(loss.data[0])    # armazena o erro
        if printout:
            print batch, loss.data[0]
    
    end = timeit.default_timer()
    elapsed = end - begin
        
    val_output = graphsage.forward(val)
    score = f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro")
    if printout:
        print "Validation F1:", score
        print "Average batch time:", np.mean(times)
    
    return train_loss, score, elapsed

def load_pubmed():
    #hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open("pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1])-1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    with open("pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists

def run_pubmed():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 19717
    feat_data, labels, adj_lists = load_pubmed()
        
    ################################################################################################################
    # sets definition earlier
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])
    
    ################################################################################################################
    # normalization
    ################################################################################################################
    scaler = preprocessing.StandardScaler().fit(feat_data[train]) # only fit in the train examples
    feat_data = scaler.transform(feat_data)
    ################################################################################################################
    ################################################################################################################

    features = nn.Embedding(19717, 500)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 500, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 10
    enc2.num_samples = 25

    graphsage = SupervisedGraphSage(3, enc2)
#    graphsage.cuda()

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
        
    ##################################################################################################
    train_loss = list()   # inicializa o vetor
    
    for batch in range(200):
        batch_nodes = train[:1024]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        ##################################################################################################
        train_loss.append(loss.data[0])    # armazena o erro
        print batch, loss.data[0]

    val_output = graphsage.forward(val) 
    score = f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro")
    print "Validation F1:", score
    print "Average batch time:", np.mean(times)
    
    return train_loss, score

if __name__ == "__main__":
    run_cora()
