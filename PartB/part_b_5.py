import numpy as np
import pandas
import tensorflow as tf
import csv
import pylab as plt

MAX_DOCUMENT_LENGTH = 100
HIDDEN_SIZE = 20
MAX_LABEL = 15
EMBEDDING_SIZE = 50
epochs = 101
batch_size = 128

N_FILTERS = 10
FILTER_SHAPE1 = [20, 256]
FILTER_SHAPE2 = [20, 1]

EMBEDDING_SIZE_COV = 20
FILTER_SHAPE1_COV = [20, 20]
FILTER_SHAPE2_COV = [20, 10]

POOLING_WINDOW = 4
POOLING_STRIDE = 2

no_epochs = 101
lr = 0.01



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

def word_cnn_model(x):

  word_vectors = tf.contrib.layers.embed_sequence(
      x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
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

def word_rnn_model(x):

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

    fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1, figsize=(6, 10))
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
    x_train_char, y_train_char, x_test_char, y_test_char = read_data_chars()
    x_train_word, y_train_word, x_test_word, y_test_word, n_words= read_data_words()

    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)

    logits_1 = char_cnn_model(x)
    logits_2 = word_cnn_model(x)
    logits_3 = char_rnn_model(x)
    logits_4 = word_rnn_model(x)

    # Optimizer
    entropy_1 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits_1))
    entropy_2 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits_2))
    entropy_3 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits_3))
    entropy_4 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits_4))

    train_op_1 = tf.train.AdamOptimizer(lr).minimize(entropy_1)
    train_op_2 = tf.train.AdamOptimizer(lr).minimize(entropy_2)
    train_op_3 = tf.train.AdamOptimizer(lr).minimize(entropy_3)
    train_op_4 = tf.train.AdamOptimizer(lr).minimize(entropy_4)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(logits_1, 1), y_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy_1 = tf.reduce_mean(correct_prediction)

    correct_prediction = tf.equal(tf.argmax(logits_2, 1), y_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy_2 = tf.reduce_mean(correct_prediction)

    correct_prediction = tf.equal(tf.argmax(logits_3, 1), y_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy_3 = tf.reduce_mean(correct_prediction)

    correct_prediction = tf.equal(tf.argmax(logits_4, 1), y_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy_4 = tf.reduce_mean(correct_prediction)

    with tf.Session()as sess:

        print("CharCNN...")
        sess.run(tf.global_variables_initializer())

        # training
        N = len(x_train_char)
        test_acc = []
        train_err = []
        for i in range(no_epochs):
            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_op_1.run(feed_dict={x: x_train_char[start:end], y_: y_train_char[start:end]})

            test_acc.append(accuracy_1.eval(feed_dict={x: x_test_char, y_: y_test_char}))
            train_err.append(entropy_1.eval(feed_dict={x: x_train_char, y_: y_train_char}))

            if i % 10 == 0:
                print('iter %d: test accuracy %g' % (i, test_acc[i]))
                print('iter %d: cross entropy %g' % (i, train_err[i]))

        plot_err_acc(train_err, test_acc)

        print("WordCNN...")
        sess.run(tf.global_variables_initializer())

        # training
        N = len(x_train_word)
        test_acc = []
        train_err = []
        for i in range(no_epochs):
            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_op_2.run(feed_dict={x: x_train_word[start:end], y_: y_train_word[start:end]})

            test_acc.append(accuracy_2.eval(feed_dict={x: x_test_word, y_: y_test_word}))
            train_err.append(entropy_2.eval(feed_dict={x: x_train_word, y_: y_train_word}))

            if i % 10 == 0:
                print('iter %d: test accuracy %g' % (i, test_acc[i]))
                print('iter %d: cross entropy %g' % (i, train_err[i]))

        plot_err_acc(train_err, test_acc)

        print("CharRNN...")
        sess.run(tf.global_variables_initializer())

        # training
        N = len(x_train_char)
        test_acc = []
        train_err = []
        for i in range(no_epochs):
            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_op_3.run(feed_dict={x: x_train_char[start:end], y_: y_train_char[start:end]})

            test_acc.append(accuracy_3.eval(feed_dict={x: x_test_char, y_: y_test_char}))
            train_err.append(entropy_3.eval(feed_dict={x: x_train_char, y_: y_train_char}))

            if i % 10 == 0:
                print('iter %d: test accuracy %g' % (i, test_acc[i]))
                print('iter %d: cross entropy %g' % (i, train_err[i]))

        plot_err_acc(train_err, test_acc)

        print("WordRNN...")
        sess.run(tf.global_variables_initializer())

        # training
        N = len(x_train_word)
        test_acc = []
        train_err = []
        for i in range(no_epochs):
            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_op_4.run(feed_dict={x: x_train_word[start:end], y_: y_train_word[start:end]})

            test_acc.append(accuracy_4.eval(feed_dict={x: x_test_word, y_: y_test_word}))
            train_err.append(entropy_4.eval(feed_dict={x: x_train_word, y_: y_train_word}))

            if i % 10 == 0:
                print('iter %d: test accuracy %g' % (i, test_acc[i]))
                print('iter %d: cross entropy %g' % (i, train_err[i]))

        plot_err_acc(train_err, test_acc)


if __name__ == '__main__':
    main()

