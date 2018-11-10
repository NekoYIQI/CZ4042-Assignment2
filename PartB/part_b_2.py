import numpy as np
import pandas
import tensorflow as tf
import csv
import pylab as plt

MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
FILTER_SHAPE1 = [20, 256]
FILTER_SHAPE2 = [20, 1]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
MAX_LABEL = 15

no_epochs = 501
batch_size = 128
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 0
tf.set_random_seed(seed)

def word_cnn_model(x):
  
  input_layer = tf.reshape(
      tf.one_hot(x, 256), [-1, MAX_DOCUMENT_LENGTH, 256, 1])

  with tf.variable_scope('CNN_Layer1'):
    conv1 = tf.layers.conv2d(
        input_layer,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE1,
        padding='VALID',
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')

  with tf.variable_scope('CNN_Layer2'):  
    conv2 = tf.layers.conv2d(
        pool1,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE2,
        padding='VALID',
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(
        conv2,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')
    pool2 = tf.squeeze(tf.reduce_max(pool2, 1), squeeze_dims=[1])

  logits = tf.layers.dense(pool2, MAX_LABEL, activation=None)

  return input_layer, logits


def read_data_chars():
  
  x_train, y_train, x_test, y_test = [], [], [], []

  with open('train_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_train.append(row[1])
      y_train.append(int(row[0]))

  with open('test_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_test.append(row[1])
      y_test.append(int(row[0]))
  
  x_train = pandas.Series(x_train)
  y_train = pandas.Series(y_train)
  x_test = pandas.Series(x_test)
  y_test = pandas.Series(y_test)
  
  
  char_processor = tf.contrib.learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
  x_train = np.array(list(char_processor.fit_transform(x_train)))
  x_test = np.array(list(char_processor.transform(x_test)))
  y_train = y_train.values
  y_test = y_test.values
  
  return x_train, y_train, x_test, y_test

def plot_err_acc(err, acc):
    num_epoch = len(err)
    
    fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1,figsize=(6, 10)) 
    ax1.plot(range(num_epoch), err)
    ax2.plot(range(num_epoch), acc)

    ax1.set_xlabel('epoch')
    ax2.set_xlabel('epoch')
    ax1.set_ylabel('Train Errors')
    ax2.set_ylabel('Test Accuracy')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

  
def main():
  
  x_train, y_train, x_test, y_test = read_data_chars()

  print(len(x_train))
  print(len(x_test))

  # Create the model
  x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
  y_ = tf.placeholder(tf.int64)

  inputs, logits = char_cnn_model(x)

  # Optimizer
  entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
  train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

  # Accuracy
  correct_prediction = tf.equal(tf.argmax(logits, 1), y_)
  correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())

    # training
    N = len(x_train)
    test_acc = []
    train_err = []
    for i in range(no_epochs):
      for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
        train_op.run(feed_dict={x: x_train[start:end], y_: y_train[start:end]})

      test_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))
      train_err.append(entropy.eval(feed_dict={x: x_train, y_: y_train}))

      if i%100 == 0:
        print('iter %d: test accuracy %g'%(i, test_acc[i]))
        print('iter %d: cross entropy %g'%(i, train_err[i]))

    plot_err_acc(train_err, test_acc)   
  

if __name__ == '__main__':
  main()
