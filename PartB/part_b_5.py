import numpy as np
import pandas
import tensorflow as tf
import csv
import time
import pylab as plt

MAX_DOCUMENT_LENGTH = 100
HIDDEN_SIZE = 20
MAX_LABEL = 15
EMBEDDING_SIZE = 50
batch_size = 128

N_FILTERS = 10
FILTER_SHAPE1 = [20, 256]
FILTER_SHAPE2 = [20, 1]

EMBEDDING_SIZE_COV = 20
FILTER_SHAPE1_COV = [20, 20]
FILTER_SHAPE2_COV = [20, 10]
model = ['CNN_char', 'CNN_word', 'RNN_char','RNN_word']
POOLING_WINDOW = 4
POOLING_STRIDE = 2

no_epochs = 51
lr = 0.01

runtime = []
acc = []
err = []


def char_cnn_model(x):
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

    return logits

def word_cnn_model(x, n_words):

  word_vectors = tf.contrib.layers.embed_sequence(
      x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE_COV)
  word_vectors = tf.expand_dims(word_vectors, 3)
  with tf.variable_scope('CNN_Layer3'):
      # Apply Convolution filtering on input sequence.
      conv1 = tf.layers.conv2d(
          word_vectors,
          filters=N_FILTERS,
          kernel_size=FILTER_SHAPE1_COV,
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
  with tf.variable_scope('CNN_Layer4'):
      # Second level of convolution filtering.
      conv2 = tf.layers.conv2d(
          pool1,
          filters=N_FILTERS,
          kernel_size=FILTER_SHAPE2_COV,
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

def char_rnn_model(x):

    with tf.variable_scope('RNN_Layer1'):
        byte_vectors = tf.one_hot(x, 256, 1., 0.)
        byte_list = tf.unstack(byte_vectors, axis=1)

        cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
        _, encoding = tf.nn.static_rnn(cell, byte_list, dtype=tf.float32)

        logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)


    return logits

def word_rnn_model(x, n_words):

    with tf.variable_scope('RNN_Layer2'):
        word_vectors = tf.contrib.layers.embed_sequence(
            x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

        word_list = tf.unstack(word_vectors, axis=1)

        cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
        _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

        logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)


    return logits


def read_data_chars():
    x_train, y_train, x_test, y_test = [], [], [], []

    with open('train_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_train.append(row[2])
            y_train.append(int(row[0]))

    with open("test_medium.csv", encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_test.append(row[2])
            y_test.append(int(row[0]))

    x_train = pandas.Series(x_train)
    y_train = pandas.Series(y_train)
    x_test = pandas.Series(x_test)
    y_test = pandas.Series(y_test)
    y_train = y_train.values
    y_test = y_test.values

    char_processor = tf.contrib.learn.preprocessing.ByteProcessor(
        MAX_DOCUMENT_LENGTH)

    x_transform_train = char_processor.fit_transform(x_train)
    x_transform_test = char_processor.transform(x_test)

    x_train = np.array(list(x_transform_train))
    x_test = np.array(list(x_transform_test))

    return x_train, y_train, x_test, y_test


def read_data_words():
    x_train, y_train, x_test, y_test = [], [], [], []

    with open('train_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_train.append(row[2])
            y_train.append(int(row[0]))

    with open("test_medium.csv", encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_test.append(row[2])
            y_test.append(int(row[0]))

    x_train = pandas.Series(x_train)
    y_train = pandas.Series(y_train)
    x_test = pandas.Series(x_test)
    y_test = pandas.Series(y_test)
    y_train = y_train.values
    y_test = y_test.values

    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        MAX_DOCUMENT_LENGTH)

    x_transform_train = vocab_processor.fit_transform(x_train)
    x_transform_test = vocab_processor.transform(x_test)

    x_train = np.array(list(x_transform_train))
    x_test = np.array(list(x_transform_test))

    no_words = len(vocab_processor.vocabulary_)
    print('Total words: %d' % no_words)

    return x_train, y_train, x_test, y_test, no_words


def plot_err_acc(err, acc, hyperparam, label):
    n = len(hyperparam)
    num_epoch = len(acc[0])

    fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1, figsize=(6, 10))
    for i in range(n):
        ax1.plot(range(num_epoch), acc[i], label='{}={}'.format(label, hyperparam[i]))
        ax2.plot(range(num_epoch), err[i], label='{}={}'.format(label, hyperparam[i]))
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Testing accuracy')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('Training error')
    ax1.legend()
    ax2.legend()
    plt.savefig('./figures/q_5.png')
    plt.show()

def NN_Model(i):

    if i == 0 or i == 2:
        x_train, y_train, x_test, y_test= read_data_chars()
    if i == 1 or i == 3:
        x_train, y_train, x_test, y_test, n_words = read_data_words()

    # Create the model
    x1 = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_1 = tf.placeholder(tf.int64)

    if i == 0:
        logits_1 = char_cnn_model(x1)
    if i == 1:
        logits_1 = word_cnn_model(x1, n_words)
    if i == 2:
        logits_1 = char_rnn_model(x1)
    if i == 3:
        logits_1 = word_rnn_model(x1, n_words)

    # Optimizer
    entropy_1 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_1, MAX_LABEL), logits=logits_1))

    train_op_1 = tf.train.AdamOptimizer(lr).minimize(entropy_1)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(logits_1, 1), y_1)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy_1 = tf.reduce_mean(correct_prediction)

    with tf.Session() as sess:


        print(model[i] + "...")

        sess.run(tf.global_variables_initializer())

        # training
        N = len(x_train)
        test_acc = []
        train_err = []
        start_time = time.time()
        for i in range(no_epochs):
            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_op_1.run(feed_dict={x1: x_train[start:end], y_1: y_train[start:end]})

            test_acc.append(accuracy_1.eval(feed_dict={x1: x_test, y_1: y_test}))
            train_err.append(entropy_1.eval(feed_dict={x1: x_train, y_1: y_train}))

            if i % 10 == 0:
                print('iter %d: test accuracy %g' % (i, test_acc[i]))
                print('iter %d: cross entropy %g' % (i, train_err[i]))
        acc.append(test_acc)
        err.append(train_err)
        run_time = time.time() - start_time
        runtime.append(run_time)


def main():

        for i in range(4):
            NN_Model(i)
        # plot test accuracy
        plot_err_acc(err, acc, model, "Model")
        # plot run time
        plt.figure(1)
        plt.plot(model, runtime)
        plt.xlabel('model')
        plt.ylabel('run time')

        plt.show()

if __name__ == '__main__':
    main()

