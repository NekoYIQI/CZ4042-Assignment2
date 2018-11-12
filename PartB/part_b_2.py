import numpy as np
import pandas
import tensorflow as tf
import csv
import pylab as plt

MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
EMBEDDING_SIZE = 20
FILTER_SHAPE1 = [20, 20]
FILTER_SHAPE2 = [20, 10]
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

  word_vectors = tf.contrib.layers.embed_sequence(
      x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
  word_vectors = tf.expand_dims(word_vectors, 3)
  with tf.variable_scope('CNN_Layer1'):
      # Apply Convolution filtering on input sequence.
      conv1 = tf.layers.conv2d(
          word_vectors,
          filters=N_FILTERS,
          kernel_size=FILTER_SHAPE1,
          padding='VALID',
          # Add a ReLU for non linearity.
          activation=tf.nn.relu)
      # Max pooling across output of Convolution+Relu.
      pool1 = tf.layers.max_pooling2d(
          conv1,
          pool_size=POOLING_WINDOW,
          strides=POOLING_STRIDE,
          padding='SAME')
      # Transpose matrix so that n_filters from convolution becomes width.
      pool1 = tf.transpose(pool1, [0, 1, 3, 2])
  with tf.variable_scope('CNN_Layer2'):
      # Second level of convolution filtering.
      conv2 = tf.layers.conv2d(
          pool1,
          filters=N_FILTERS,
          kernel_size=FILTER_SHAPE2,
          padding='VALID')
      # Max across each filter to get useful features for classification.
      pool2 = tf.layers.max_pooling2d(
          conv2,
          pool_size=POOLING_WINDOW,
          strides=POOLING_STRIDE,
          padding='SAME'
      )
      pool2 = tf.squeeze(tf.reduce_max(pool2, 1), axis=[1])

  # Apply regular WX + B and classification.
  logits = tf.layers.dense(pool2, MAX_LABEL, activation=None)

  return logits


def read_data_words():
  
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
  
  
  vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
  x_train = np.array(list(vocab_processor.fit_transform(x_train)))
  x_test = np.array(list(vocab_processor.transform(x_test)))
  y_train = y_train.values
  y_test = y_test.values

  no_words = len(vocab_processor.vocabulary_)
  print('Total words: %d' % no_words)
  
  return x_train, y_train, x_test, y_test, no_words

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
  global n_words
  x_train, y_train, x_test, y_test, n_words = read_data_words()

  print(len(x_train))
  print(len(x_test))

  # Create the model
  x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
  y_ = tf.placeholder(tf.int64)

  logits = word_cnn_model(x)

  # Optimizer
  entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
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
