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

import sys
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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
    ######################################################################################################################
    #num_feats = 1433  # acrescentando 3 colunas para as features do grafo
    num_feats = 1436
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
            # cria a matriz de features retirando o primeiro e ultimo elemento, idPaper e Classe respectivamente
            # o que sobra sao as informacoes de presenca das words (0,1)
            feat_data[i,:-3] = map(float, info[1:-1])
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    ######################################################################################################################
    # acrescenta as features do grafo
    ######################################################################################################################
    with open("cora/cora_graph.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[node_map[info[0]],-3:] = map(float, info[1:])
            
    #print(feat_data[node_map['35'],-3:])
    #print(feat_data[node_map['4330'],-3:])
    #print(feat_data[node_map['312409'],-3:])
    #print(feat_data[node_map['1154074'],-3:])
    ######################################################################################################################
    ######################################################################################################################

    adj_lists = defaultdict(set)
    with open("cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
            
            
    return feat_data, labels, adj_lists


def run_cora():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708
    feat_data, labels, adj_lists = load_cora()
    
    ######################################################################################################################
    # TESTAR AS DUAS FORMAS: INCLUIR NA MATRIZ DE FEATURES AS INFORMACOES
    #                        (talvez isso nao seja necessario ja que o graphsage faz os embeddings por agregacao)
    #                                                    OU
    #                        APENAS USAR O RESULTADO DO K-MEANS PARA SEPARAR OS LOTES
    ######################################################################################################################

    # normalization
    scaler = preprocessing.StandardScaler().fit(feat_data)
    feat_data = scaler.transform(feat_data)
    
    # Metodo Elbow para encontrar o numero de classes
    if False:
        wcss = [] # Within Cluster Sum of Squares
        for i in range(2, 11):
            kmeans = KMeans(n_clusters = i, init = 'random')
            kmeans.fit(feat_data)
            print i,kmeans.inertia_
            wcss.append(kmeans.inertia_)  
        plt.plot(range(2, 11), wcss)
        plt.title('O Metodo Elbow')
        plt.xlabel('Numero de Clusters')
        plt.ylabel('WCSS')
        plt.show()
        #return None, None
        
    
    # Metodo Elbow encontrou 7 classes
    kmeans = KMeans(n_clusters = 7, init = 'random')
    kmeans.fit(feat_data)
    kmeans.fit_transform(feat_data)
    klabels = kmeans.labels_
    if False:
        return labels, klabels
    
    ######################################################################################################################
    ######################################################################################################################
    
    
    ######################################################################################################################
    #features = nn.Embedding(2708, 1433)  # acrescentando 3 colunas para as features do grafo
    features = nn.Embedding(2708, 1436)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    # MeanAggregator params
    #  features, cuda=False, gcn=False

    # Encoder params
    #  features, feature_dim, embed_dim, adj_lists, aggregator, num_sample=10, base_model=None, 
    #  gcn=False, cuda=False, feature_transform=False
    
    agg1 = MeanAggregator(features, cuda=True)
    ######################################################################################################################
    #enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=False)  # acrescentando 3 colunas para as features do grafo
    enc1 = Encoder(features, 1436, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(7, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes) # len(rand_indices) = 2708
    
    ###################################################################################################################
    # AQUI EU ATRIBUO INDICES AOS CONJUNTOS USANDO AS QUANTIDADES PROPORCIONAIS DE CADA K-MEANS CLUSTER
    ###################################################################################################################
    #test = rand_indices[:1000]        # 1000 exemplos
    #val = rand_indices[1000:1500]     #  500 exemplos
    #train = list(rand_indices[1500:]) # 1208 exemplos
    
    train, val, test, ratio = _processes_set(klabels)
    
    ###################################################################################################################
    ###################################################################################################################

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    
    ###################################################################################################################
    # quantidade proporcional do batch, inicializacao do vetor de erro
    ###################################################################################################################
    quantity = np.empty((7,1), dtype=int)
    quantity[:,0] = ratio * 256
    quantity = list(quantity.flatten())
    
    train_loss = list()
    ###################################################################################################################
    ###################################################################################################################
    
    for batch in range(100):
        ##################################################################################################
        # O QUE EU POSSO FAZER AQUI EH EMBARALHAR OS VERTICES SEPARADAMENTE DENTRO DOS CLUSTERS
        # PARA MONTAR BATCH_NODES, PEGAR AS QUANTIDADES PROPORCIONAIS 
        ##################################################################################################
        batch_nodes = list()
        for key in train:
            batch_nodes.extend(train[key][:quantity[key]])
            random.shuffle(train[key])
        random.shuffle(batch_nodes)
        ##################################################################################################
        ##################################################################################################
        
        # pega os primeiros 255 exemplos
        #batch_nodes = train[:256]
        # embaralha os exemplos para que o proximo batch_nodes tenha outros vertices
        # a amostragem, nesse caso, eh com reposicao
        #random.shuffle(train)
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

def _processes_set(klabels):
    
    #### separa os indices por cluster
    klabels_map = dict()
    for i in range(7):
        klabels_map[i] = list()
    for i, label in enumerate(klabels):
        klabels_map[label].append(i)
    
    
    #### embaralha os indices
    for i in range(7):
        random.shuffle(klabels_map[i])
    
    
    #### separa os conjuntos
    # calcula a razao
    ratio = np.array([len(klabels_map[key]) / float(2708) for key in klabels_map])

    # calcula quantidade proporcional dos conjuntos: treino(0), val(1), teste(2)
    quantity = np.empty((7,3), dtype=int)
    quantity[:,0] = ratio * 1208
    quantity[:,1] = ratio * 500
    quantity[:,2] = ratio * 1000
    
    # distribui os indices proporcionalmente por conjunto
    val = list()
    test = list()
    train = dict()
    
    for i in range(7):
        train[i] = list()

    for key in klabels_map:
        limit_train_quantity = quantity[key,0]
        train[key] = klabels_map[key][:limit_train_quantity]
        limit_val_quantity = limit_train_quantity + quantity[key,1]
        val.extend(klabels_map[key][limit_train_quantity:limit_val_quantity])
        test.extend(klabels_map[key][limit_val_quantity:])
        
    return train, val, test, ratio

    
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
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
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
        print batch, loss.data[0]

    val_output = graphsage.forward(val) 
    print "Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro")
    print "Average batch time:", np.mean(times)

    
if __name__ == "__main__":
    run_cora()
