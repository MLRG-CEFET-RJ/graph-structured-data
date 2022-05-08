import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import networkx as nx
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.python.ops import gen_array_ops
from PIL import Image

NUMBER_NODES = 9

def load_data():
    train_df = pd.read_csv(os.path.join('..', 'DNN', 'datasets', f'dataset_{NUMBER_NODES}_train.csv'))
    val_df = pd.read_csv(os.path.join('..', 'DNN', 'datasets', f'dataset_{NUMBER_NODES}_val.csv'))
    test_df = pd.read_csv(os.path.join('..', 'DNN', 'datasets', f'dataset_{NUMBER_NODES}_test.csv'))

    featuresNumber = (NUMBER_NODES * NUMBER_NODES - NUMBER_NODES) // 2 
    def get_tuple_tensor_dataset(row):
        X = row[0 : featuresNumber].astype('float32')
        Y = row[featuresNumber + 1: ].astype('float32') # Pula a banda otima na posicao 0
        return X, Y

    train_dataset = list(map(get_tuple_tensor_dataset, train_df.to_numpy()))
    val_dataset = list(map(get_tuple_tensor_dataset, val_df.to_numpy()))
    test_dataset = list(map(get_tuple_tensor_dataset, test_df.to_numpy()))

    X = []
    Y = []
    for x, y in train_dataset:
        X.append(x)
        Y.append(y)
    x_train = np.array(X)
    y_train = np.array(Y)

    X = []
    Y = []
    for x, y in test_dataset:
        X.append(x)
        Y.append(y)
    x_test = np.array(X)
    y_test = np.array(Y)

    X = []
    Y = []
    for x, y in val_dataset:
        X.append(x)
        Y.append(y)
    x_val = np.array(X)
    y_val = np.array(Y)

    x_train = np.concatenate((x_train, x_val))
    y_train = np.concatenate((y_train, y_val))


    return x_train, y_train, x_test, y_test

X, y, x_t, y_t = load_data()

print(X.shape)
print(y.shape)

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

def processDataToAdjImage(graphInput):
    adj = getGraph(graphInput)
    w, h = NUMBER_NODES, NUMBER_NODES
    data = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i, j] == 1:
                data[i, j] = np.array([255.0, 255.0, 255.0])
    # data /= 255.0
    img = Image.fromarray(data, 'RGB')
    resized = img.resize((32, 32), Image.NEAREST)
    image_input_np = np.array(resized)
    return image_input_np

def getData_2(features, labels):
    train_images = []
    train_nodelist = []
    for graphInput, target in zip(features, labels):
        graphNodeList = target
        x_image = processDataToAdjImage(graphInput)
        train_images.append(x_image)
        train_nodelist.append(graphNodeList)
    # mlb = MultiLabelBinarizer()
    # labels = mlb.fit_transform(train_nodelist)
    return np.array(train_images), np.array(train_nodelist)

X_train, y_train = getData_2(X, y)
x_test, y_test = getData_2(x_t, y_t)

print(X_train.shape)
print(y_train.shape)

it = iter(X_train)
entry = next(it)
entry

plt.imshow(entry)
plt.savefig('an_entry_9vertices.jpg')
plt.clf()

mseLoss = tf.keras.losses.MeanSquaredError()

def loss_repeated_labels(roundedOutput):
  # true_used, true_indexes = tf.unique(tf.squeeze(true))
  used_labels, indexes, counts = tf.unique_with_counts(tf.squeeze(roundedOutput))
  counts = tf.cast(counts, tf.float32)
  # 1 - counts = quao longe os elementos de counts estão de repetir uma vez só (elemento unico)
  mse_ones_like = mseLoss(tf.ones_like(counts), counts) 
  # mseIndexes = loss_object(tf.cast(true_indexes, tf.float32), tf.cast(indexes, tf.float32))

  variance = tf.math.reduce_variance(counts)
  return mse_ones_like + variance

def customLoss(true, pred):
  mse = mseLoss(true, pred)
  roundedOutput = tf.round(pred)
  loss_repeated = loss_repeated_labels(roundedOutput)
  return mse + loss_repeated


data_augmentation = tf.keras.Sequential(
  [
    # layers.RandomFlip("horizontal", input_shape=(32, 32, 3)),
    layers.RandomRotation(0.2),
  ]
)

model = tf.keras.models.Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.3),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(NUMBER_NODES)
])

model.compile(optimizer='adam',
              loss=customLoss,
              metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    validation_data=(x_test, y_test),
    epochs=128,
    batch_size=1
)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig('accuracy_9vertices.jpg')


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.savefig('loss_9vertices.jpg')

def count_repeats(output):
    counts = np.unique(np.round(output))
    repeated = NUMBER_NODES - counts.shape[0]
    return repeated

def get_valid_pred(pred):
    valid = np.ones(NUMBER_NODES)
    labels = np.arange(0, NUMBER_NODES)
    for i in labels:
        min_value = np.amin(pred)
        min_idx = np.where(pred == min_value)
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

def get_array_from_image(graphnp):
    img = Image.fromarray(graphnp, 'RGB')
    img = img.convert('L')
    resized = img.resize((NUMBER_NODES, NUMBER_NODES), Image.NEAREST)
    image_input_np = np.array(resized)
    return image_input_np / 255

import time

pred = model.predict(x_test)

sumTest_original = []
sumTest_pred = []
sumTest_true = []

count = 0
cases_with_repetition = 0

start = time.time()
for i in range(len(pred)):
    output = pred[i]
    quantity_repeated = count_repeats(np.round(output))
    print('Pred: ', output)
    print('True: ', y_test[i])
    if quantity_repeated != 0:
        cases_with_repetition += 1
    output = get_valid_pred(output)
    print('Pred valid: ', output)
    count += quantity_repeated

    graph = get_array_from_image(x_test[i])

    original_band = get_bandwidth(graph, np.array(None))
    sumTest_original.append(original_band)

    pred_band = get_bandwidth(graph, output)
    sumTest_pred.append(pred_band)

    true_band = get_bandwidth(graph, y_test[i])
    sumTest_true.append(true_band)

    # print("Bandwidth")
    # print(original_band)
    # print(pred_band)
    # print(true_band)
end = time.time()

print('Quantidade de rótulos repetidos, exemplo [1, 1, 1, 1, 1, 1, 1] conta como 6 - ', count)
print('Quantidade de saídas com repetição, exemplo [1, 1, 1, 1, 1, 1, 1] conta como 1 - ', cases_with_repetition)
test_length = pred.shape[0]
print('Test length - ', test_length)
print('Time- ', (end - start) / test_length)

print("Bandwidth mean±std")
print(f'{np.mean(sumTest_original)}±{np.std(sumTest_original)}')
print("Pred bandwidth mean±std")
print(f'{np.mean(sumTest_pred)}±{np.std(sumTest_pred)}')
print("True bandwidth mean±std")
print(f'{np.mean(sumTest_true)}±{np.std(sumTest_true)}')