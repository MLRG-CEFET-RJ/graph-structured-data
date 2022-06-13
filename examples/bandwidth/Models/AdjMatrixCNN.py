import os
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from PIL import Image
from ModelInterface import ModelInterface
from Helper import Helper
import argparse
import numpy as np
import networkx as nx
import time
from tensorflow.keras.models import load_model

class CNNHelper(Helper):
  def __init__(self, NUMBER_NODES):
    super().__init__(NUMBER_NODES)

  def process_data_to_image_np_array(self, graphInput):
    adj = super().getGraph(graphInput)
    w, h = self.NUMBER_NODES, self.NUMBER_NODES
    data = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i, j] == 1:
                data[i, j] = np.array([255.0, 255.0, 255.0])
    img = Image.fromarray(data, 'RGB')
    resized = img.resize((self.NUMBER_NODES * 4, self.NUMBER_NODES * 4), Image.NEAREST)
    image_input_np = np.array(resized)
    return image_input_np

  def get_image_dataset(self, features, labels):
      train_images = []
      train_nodelist = []
      for graphInput, target in zip(features, labels):
          graphNodeList = target
          x_image = self.process_data_to_image_np_array(graphInput)
          train_images.append(x_image)
          train_nodelist.append(graphNodeList)
      return np.array(train_images), np.array(train_nodelist)

  def get_matrix_from_image(self, graphnp):
    img = Image.fromarray(graphnp, 'RGB')
    img = img.convert('L')
    resized = img.resize((self.NUMBER_NODES, self.NUMBER_NODES), Image.NEAREST)
    image_input_np = np.array(resized)
    return image_input_np / 255

class LossRepeatedLabels(tf.keras.losses.Loss):
  def __init__(self):
    super().__init__()
    self.mseInstance = tf.keras.losses.MeanSquaredError()

  def loss_repeated_labels(self, roundedOutput):
    def body(i, acc, out):
      used_labels, indexes, counts = tf.unique_with_counts(out[i])
      counts = tf.cast(counts, tf.float32)
      # variance = tf.math.reduce_variance(counts)
      # return (i + 1, tf.add(acc, variance), out)
      counts_shape = tf.shape(counts)[0]
      mse = self.mseInstance(tf.ones(counts_shape), counts)
      return (i + 1, tf.add(acc, mse), out)
  
    acc = tf.constant(0, dtype=tf.float32)
    out = roundedOutput
    batch_size = tf.shape(out)[0]

    i = tf.constant(0)
    condition = lambda i, acc, out: tf.less(i, batch_size)
    b = body
    result = tf.while_loop(condition, b, loop_vars=[i, acc, out])
    return result

  def call(self, y_true, y_pred):
    mse = self.mseInstance(y_true, y_pred)
    roundedOutput = tf.round(y_pred)
    i, loss_repeated, roundedOutput = self.loss_repeated_labels(roundedOutput)
    return mse + loss_repeated

class AdjMatrixCNN(ModelInterface):
  def __init__(self, NUMBER_NODES, batch_size, epochs):
    self.NUMBER_NODES = NUMBER_NODES
    self.batch_size = batch_size
    self.epochs = epochs

  def fit(self):
    data_augmentation = tf.keras.Sequential(
      [
        layers.RandomRotation(0.2),
        layers.RandomContrast(0.2),
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
      layers.Dense(self.NUMBER_NODES)
    ])

    model.compile(optimizer='adam',
                  loss=LossRepeatedLabels(),
                  metrics=['accuracy'])

    x_train, y_train = super().load_train_data(datatype='int32', NUMBER_NODES=self.NUMBER_NODES)
    x_test, y_test = super().load_test_data(datatype='int32', NUMBER_NODES=self.NUMBER_NODES)

    helper = CNNHelper(self.NUMBER_NODES)

    x_train, y_train = helper.get_image_dataset(x_train, y_train)
    x_test, y_test = helper.get_image_dataset(x_test, y_test)

    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=self.epochs,
        batch_size=self.batch_size,
        verbose=2
    )

    x_train = None
    y_train = None
    x_test = None
    y_test = None
    # helps garbage colleciton

    if not os.path.exists('plotted_figures'):
      os.makedirs('plotted_figures')

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join('plotted_figures', f'AdjMatrixCNN_accuracy_{self.NUMBER_NODES}_vertices.jpg'))
    plt.clf()

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join('plotted_figures', f'AdjMatrixCNN_loss_{self.NUMBER_NODES}_vertices.jpg'))
    plt.clf()

    if not os.path.exists('saved_models'):
      os.makedirs('saved_models')

    model.save(
      os.path.join('saved_models', f'AdjMatrixCNN_{self.NUMBER_NODES}_vertices.h5'),
      save_format='h5'
    )
  def predict(self):
    try:
      model = load_model(
        os.path.join('saved_models', f'AdjMatrixCNN_{self.NUMBER_NODES}_vertices.h5'),
        compile=False
      )

      x_test, y_test = super().load_test_data(datatype='int32', NUMBER_NODES=self.NUMBER_NODES)
      helper = CNNHelper(self.NUMBER_NODES)
      x_test, y_test = helper.get_image_dataset(x_test, y_test)

      x_dataset = tf.data.Dataset.from_tensor_slices(x_test)
      x_dataset = x_dataset.batch(32)

      y_dataset = tf.data.Dataset.from_tensor_slices(y_test)
      y_dataset = y_dataset.batch(32)

      sumTest_original = []
      sumTest_pred = []
      sumTest_true = []
      prediction_times = []

      count = 0
      cases_with_repetition = 0

      test_length = x_test.shape[0]
      print(test_length)

      for x_batch, y_batch in zip(x_dataset, y_dataset):
        start_time = time.time()

        output_batch = model.predict(x_batch, verbose=2)

        output_batch, quantity_repeated, cases_repeated = helper.get_valid_preds(output_batch)
        count += quantity_repeated
        cases_with_repetition += cases_repeated

        prediction_times.append(time.time() - start_time)

        for features, pred, target in zip(x_batch, output_batch, y_batch):
          graph = helper.get_matrix_from_image(features.numpy())
          graph = nx.Graph(graph)
          
          original_band = helper.get_bandwidth(graph, np.array(None))
          sumTest_original.append(original_band)

          pred_band = helper.get_bandwidth(graph, pred)
          sumTest_pred.append(pred_band)

          true_band = helper.get_bandwidth(graph, target.numpy())
          sumTest_true.append(true_band)

      x_test = None
      y_test = None

      AdjMatrixCNNResult = helper.getResult(
        model_name='AdjMatrixCNN',
        sumTest_original=sumTest_original,
        sumTest_pred=sumTest_pred,
        sumTest_true=sumTest_true,
        count=count,
        cases_with_repetition=cases_with_repetition,
        prediction_times=prediction_times
      )
      return AdjMatrixCNNResult
    except FileNotFoundError as e:
      print(e)
      print('Error loading the model from disk, should run model.fit')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Tensorflow - Convolutional Neural Network')
  parser.add_argument('-v','--vertices', help='number of vertices dataset [7, 9]', required=True)
  parser.add_argument('-m','--mode', help='0 - fit, 1 - predict', required=True)
  parser.add_argument('-b','--batch', help="batch_size - 32, 64, 128, 256, ...", required=True)
  parser.add_argument('-e','--epochs', help="epochs - 200, 256, ...", required=True)
  args = parser.parse_args()

  NUMBER_NODES = int(args.vertices)
  batch_size = int(args.batch)
  epochs = int(args.epochs)

  adjMatrixCNN = AdjMatrixCNN(NUMBER_NODES=NUMBER_NODES, batch_size=batch_size, epochs=epochs)

  if args.mode == '0':
    adjMatrixCNN.fit()
  if args.mode == '1':
    df_result = adjMatrixCNN.predict()
    print(df_result.to_latex())
