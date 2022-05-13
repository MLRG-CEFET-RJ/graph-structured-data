import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from PIL import Image
from ModelInterface import ModelInterface
from Helper import Helper
import argparse
import numpy as np
import time
from tensorflow.keras.models import load_model

class HelperEspecialization(Helper):
  def __init__(self, NUMBER_NODES):
    super().__init__(NUMBER_NODES)

  def process_data_to_image(self, graphInput):
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

  def get_image_data(self, features, labels):
      train_images = []
      train_nodelist = []
      for graphInput, target in zip(features, labels):
          graphNodeList = target
          x_image = self.process_data_to_image(graphInput)
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
    super().__init__(NUMBER_NODES)
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
      layers.Dense(NUMBER_NODES)
    ])

    batch_size = self.batch_size

    model.compile(optimizer='adam',
                  loss=LossRepeatedLabels(),
                  metrics=['accuracy'])

    x_train, y_train = super().load_train_data(datatype='int32')
    x_test, y_test = super().load_test_data(datatype='int32')

    helper = HelperEspecialization(self.NUMBER_NODES)

    x_train, y_train = helper.get_image_data(x_train, y_train)
    x_test, y_test = helper.get_image_data(x_test, y_test)

    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=self.epochs,
        batch_size=self.batch_size,
    )

    if not os.path.exists('plotted_figures'):
      os.makedirs('plotted_figures')

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(f'plotted_figures/AdjMatrixCNN_accuracy_{self.NUMBER_NODES}_vertices.jpg')
    plt.clf()

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.savefig(f'plotted_figures/AdjMatrixCNN_loss_{self.NUMBER_NODES}_vertices.jpg')
    plt.clf()

    if not os.path.exists('saved_models'):
      os.makedirs('saved_models')

    model.save('saved_models/AdjMatrixCNN.h5', save_format='h5')
  def predict(self):
    try:
      model = load_model('saved_models/AdjMatrixCNN.h5', compile=False)

      x_test, y_test = super().load_test_data(datatype='int32')
      helper = HelperEspecialization(self.NUMBER_NODES)
      x_test, y_test = helper.get_image_data(x_test, y_test)

      pred = model.predict(x_test)

      sumTest_original = []
      sumTest_pred = []
      sumTest_true = []

      count = 0
      cases_with_repetition = 0

      start_time = time.time()
      for i in range(len(pred)):
          output = pred[i]

          quantity_repeated = helper.count_repeats(np.round(output))

          if quantity_repeated != 0:
              cases_with_repetition += 1
          count += quantity_repeated

          output = helper.get_valid_pred(output)

          graph = helper.get_matrix_from_image(x_test[i])
          
          original_band = helper.get_bandwidth(graph, np.array(None))
          sumTest_original.append(original_band)

          pred_band = helper.get_bandwidth(graph, output)
          sumTest_pred.append(pred_band)

          true_band = helper.get_bandwidth(graph, y_test[i])
          sumTest_true.append(true_band)
      end_time = time.time()

      test_length = pred.shape[0]
      print(test_length)

      AdjMatrixCNNResult = helper.getResult(
        model_name='AdjMatrixCNNResult',
        sumTest_original=sumTest_original,
        sumTest_pred=sumTest_pred,
        sumTest_true=sumTest_true,
        count=count,
        cases_with_repetition=cases_with_repetition,
        mean_time=(end_time - start_time) / test_length
      )
      return AdjMatrixCNNResult
    except FileNotFoundError as e:
      print(e)
      print('Error loading the model from disk, should run model.fit')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Pytorch - Deep learning custom loss')
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