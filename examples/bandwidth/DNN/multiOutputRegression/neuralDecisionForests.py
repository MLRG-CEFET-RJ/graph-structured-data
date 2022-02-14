# %%
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import math
import os
import numpy as np
import networkx as nx

# %%
class NeuralDecisionTree(keras.Model):
    def __init__(self, depth, num_features, used_features_rate, num_classes):
        super(NeuralDecisionTree, self).__init__()
        self.depth = depth
        self.num_leaves = 2 ** depth
        self.num_classes = num_classes

        # Create a mask for the randomly selected features.
        num_used_features = int(num_features * used_features_rate)
        one_hot = np.eye(num_features)
        sampled_feature_indicies = np.random.choice(
            np.arange(num_features), num_used_features, replace=False
        )
        self.used_features_mask = one_hot[sampled_feature_indicies]

        # Initialize the weights of the classes in leaves.
        self.pi = tf.Variable(
            initial_value=tf.random_normal_initializer()(
                shape=[self.num_leaves, self.num_classes]
            ),
            dtype="float32",
            trainable=True,
        )

        # Initialize the stochastic routing layer.
        self.decision_fn = layers.Dense(
            # units=self.num_leaves, activation="sigmoid", name="decision"
            units=self.num_leaves, activation="relu", name="decision"
        )

    def call(self, features):
        batch_size = tf.shape(features)[0]

        features = tf.matmul(
            features, self.used_features_mask, transpose_b=True
        )  
        
        decisions = tf.expand_dims(
            self.decision_fn(features), axis=2
        )  
        
        decisions = layers.concatenate(
            [decisions, 1 - decisions], axis=2
        )  # [batch_size, num_leaves, 2]

        mu = tf.ones([batch_size, 1, 1])

        begin_idx = 1
        end_idx = 2
        # Traverse the tree in breadth-first order.
        for level in range(self.depth):
            mu = tf.reshape(mu, [batch_size, -1, 1])  # [batch_size, 2 ** level, 1]
            mu = tf.tile(mu, (1, 1, 2))  # [batch_size, 2 ** level, 2]
            level_decisions = decisions[
                :, begin_idx:end_idx, :
            ]  # [batch_size, 2 ** level, 2]
            mu = mu * level_decisions  # [batch_size, 2**level, 2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (level + 1)

        mu = tf.reshape(mu, [batch_size, self.num_leaves])  # [batch_size, num_leaves]
        # probabilities = keras.activations.softmax(self.pi)  # [num_leaves, num_classes]
        # probabilities = keras.activations.relu(self.pi)  # [num_leaves, num_classes] - ate agr o menos errado
        outputs = tf.matmul(mu, self.pi)  # [batch_size, num_classes]
        return outputs

# %%
NUMBER_NODES = 9

def get_train_dataset():
    train_df = pd.read_csv(os.path.join('..', 'datasets', f'dataset_{NUMBER_NODES}_train.csv'))
    val_df = pd.read_csv(os.path.join('..', 'datasets', f'dataset_{NUMBER_NODES}_val.csv'))

    featuresNumber = (NUMBER_NODES * NUMBER_NODES - NUMBER_NODES) // 2 
    def get_tuple_tensor_dataset(row):
        X = row[0 : featuresNumber].astype('float32')
        Y = row[featuresNumber + 1: ].astype('float32') # Inclui a banda otima na posicao 0
        return X, Y

    train_dataset = list(map(get_tuple_tensor_dataset, train_df.to_numpy()))
    val_dataset = list(map(get_tuple_tensor_dataset, val_df.to_numpy()))

    X = []
    Y = []
    for x, y in train_dataset:
        X.append(x)
        Y.append(y)
    x_train = np.array(X)
    y_train = np.array(Y)

    X = []
    Y = []
    for x, y in val_dataset:
        X.append(x)
        Y.append(y)
    x_val = np.array(X)
    y_val = np.array(Y)

    x_train = np.concatenate((x_train, x_val))
    y_train = np.concatenate((y_train, y_val))

    return x_train, y_train

def get_test_dataset():
    test_df = pd.read_csv(os.path.join('..', 'datasets', f'dataset_{NUMBER_NODES}_test.csv'))

    featuresNumber = (NUMBER_NODES * NUMBER_NODES - NUMBER_NODES) // 2 
    def get_tuple_tensor_dataset(row):
        X = row[0 : featuresNumber].astype('int32')
        Y = row[featuresNumber + 1: ].astype('float32') # Inclui a banda otima na posicao 0
        return X, Y

    test_dataset = list(map(get_tuple_tensor_dataset, test_df.to_numpy()))

    X = []
    Y = []
    for x, y in test_dataset:
        X.append(x)
        Y.append(y)
    x_test = np.array(X)
    y_test = np.array(Y)


    return x_test, y_test

# %%
learning_rate = 0.01
batch_size = 32
num_epochs = 180

def loss_fn(targets, outputs):
    return tf.sqrt(tf.reduce_mean((targets - outputs)**2))

def run_experiment(model):

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['accuracy'],
    )

    x_train, y_train = get_train_dataset()

    model.fit(x=x_train, y=y_train, epochs=num_epochs)

# %%
depth = 10
used_features_rate = 1.0
# num_classes = 7
num_classes = 9

shape_input = (NUMBER_NODES * NUMBER_NODES - NUMBER_NODES) // 2

def create_tree_model():
    inputs = tf.keras.Input(shape=(shape_input,), dtype=tf.float32)
    # features = encode_inputs(inputs)
    # features = layers.BatchNormalization()(inputs)
    # num_features = features.shape[1]
    num_features = inputs.shape[1]

    tree = NeuralDecisionTree(depth, num_features, used_features_rate, num_classes)

    outputs = tree(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


tree_model = create_tree_model()
run_experiment(tree_model)


# %%
def count_repeats(output):
    counts = np.unique(np.round(output))
    repeated = NUMBER_NODES - counts.shape[0]
    return repeated

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
x, y = get_test_dataset()
pred = tree_model.predict(x)

sumTest_original = 0
sumTest_pred = 0
sumTest_true = 0

count = 0
cases_with_repetition = 0

for i in range(len(pred)):
    output = pred[i]

    quantity_repeated = count_repeats(np.round(output))
    print('Pred: ', output)
    print('True: ', y[i])
    if quantity_repeated != 0:
        cases_with_repetition += 1
    output = get_valid_pred(output)
    print('Pred valid: ', output)
    count += quantity_repeated

    print("Bandwidth")
    graph = getGraph(x[i])
    original_band = get_bandwidth(graph, np.array(None))
    sumTest_original += original_band
    pred_band = get_bandwidth(graph, output)
    sumTest_pred += pred_band
    true_band = get_bandwidth(graph, y[i])
    sumTest_true += true_band
    print("Bandwidth")
    print(original_band)
    print(pred_band)
    print(true_band)
print('Quantidade de rótulos repetidos, exemplo [1, 1, 1, 1, 1, 1, 1] conta como 6 - ', count)
print('Quantidade de saídas com repetição, exemplo [1, 1, 1, 1, 1, 1, 1] conta como 1 - ', cases_with_repetition)
test_length = pred.shape[0]
print('Test length - ', test_length)
print("Bandwidth mean")
print(sumTest_original / test_length)
print("Pred bandwidth mean")
print(sumTest_pred / test_length)
print("True bandwidth mean")
print(sumTest_true / test_length)

# %%
class NeuralDecisionForest(keras.Model):
    def __init__(self, num_trees, depth, num_features, used_features_rate, num_classes):
        super(NeuralDecisionForest, self).__init__()
        self.ensemble = []
        # Initialize the ensemble by adding NeuralDecisionTree instances.
        # Each tree will have its own randomly selected input features to use.
        for _ in range(num_trees):
            self.ensemble.append(
                NeuralDecisionTree(depth, num_features, used_features_rate, num_classes)
            )

    def call(self, inputs):
        # Initialize the outputs: a [batch_size, num_classes] matrix of zeros.
        batch_size = tf.shape(inputs)[0]
        outputs = tf.zeros([batch_size, num_classes])

        # Aggregate the outputs of trees in the ensemble.
        for tree in self.ensemble:
            outputs += tree(inputs)
        # Divide the outputs by the ensemble size to get the average.
        outputs /= len(self.ensemble)
        return outputs

# %%
raise ValueError("eaeaeaeae")

# %%
num_trees = 50
used_features_rate = 1.0
depth = 6  

def create_forest_model():
    inputs = tf.keras.Input(shape=(shape_input,), dtype=tf.float32)
    # features = layers.BatchNormalization()(inputs)
    num_features = inputs.shape[1]

    forest_model = NeuralDecisionForest(num_trees, depth, num_features, used_features_rate, num_classes)

    outputs = forest_model(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

forest_model = create_forest_model()

run_experiment(forest_model)

# %%
x, y = get_test_dataset()
pred = forest_model.predict(x)

sumTest_original = 0
sumTest_pred = 0
sumTest_true = 0

count = 0
cases_with_repetition = 0

for i in range(len(pred)):
    output = pred[i]

    quantity_repeated = count_repeats(np.round(output))
    print('Pred: ', output)
    print('True: ', y[i])
    if quantity_repeated != 0:
        cases_with_repetition += 1
    output = get_valid_pred(output)
    print('Pred valid: ', output)
    count += quantity_repeated

    print("Bandwidth")
    graph = getGraph(x[i])
    original_band = get_bandwidth(graph, np.array(None))
    sumTest_original += original_band
    pred_band = get_bandwidth(graph, output)
    sumTest_pred += pred_band
    true_band = get_bandwidth(graph, y[i])
    sumTest_true += true_band
    print("Bandwidth")
    print(original_band)
    print(pred_band)
    print(true_band)
print('Quantidade de rótulos repetidos, exemplo [1, 1, 1, 1, 1, 1, 1] conta como 6 - ', count)
print('Quantidade de saídas com repetição, exemplo [1, 1, 1, 1, 1, 1, 1] conta como 1 - ', cases_with_repetition)
test_length = pred.shape[0]
print('Test length - ', test_length)
print("Bandwidth mean")
print(sumTest_original / test_length)
print("Pred bandwidth mean")
print(sumTest_pred / test_length)
print("True bandwidth mean")
print(sumTest_true / test_length)

# %% [markdown]
# ## Resultados sem batch normalization
# 
# ### DecisionTree:
# Quantidade de rótulos repetidos, exemplo [1, 1, 1, 1, 1, 1, 1] conta como 6 -  219
# Quantidade de saídas com repetição, exemplo [1, 1, 1, 1, 1, 1, 1] conta como 1 -  63
# Test length -  63
# Bandwidth mean
# 5.904761904761905
# Pred bandwidth mean
# 4.809523809523809
# True bandwidth mean
# 3.1904761904761907
# 
# ### DecisionForest:
# Quantidade de rótulos repetidos, exemplo [1, 1, 1, 1, 1, 1, 1] conta como 6 -  127
# Quantidade de saídas com repetição, exemplo [1, 1, 1, 1, 1, 1, 1] conta como 1 -  59
# Test length -  63
# Bandwidth mean
# 5.904761904761905
# Pred bandwidth mean
# 4.777777777777778
# True bandwidth mean
# 3.1904761904761907
# 
# 
# ## Resultados com batch normalization
# 
# ### DecisionTree:
# Quantidade de rótulos repetidos, exemplo [1, 1, 1, 1, 1, 1, 1] conta como 6 -  170
# Quantidade de saídas com repetição, exemplo [1, 1, 1, 1, 1, 1, 1] conta como 1 -  63
# Test length -  63
# Bandwidth mean
# 5.904761904761905
# Pred bandwidth mean
# 4.936507936507937
# True bandwidth mean
# 3.1904761904761907
# 
# ### DecisionForest:
# Quantidade de rótulos repetidos, exemplo [1, 1, 1, 1, 1, 1, 1] conta como 6 -  147
# Quantidade de saídas com repetição, exemplo [1, 1, 1, 1, 1, 1, 1] conta como 1 -  60
# Test length -  63
# Bandwidth mean
# 5.904761904761905
# Pred bandwidth mean
# 4.904761904761905
# True bandwidth mean
# 3.1904761904761907


