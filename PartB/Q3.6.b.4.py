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

lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)


def rnn_model(x):
    word_vectors = tf.contrib.layers.embed_sequence(
         x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
    word_list = tf.unstack(word_vectors, axis=1)

    #byte_vectors = tf.one_hot(x, 256, 1., 0.)
    #byte_list = tf.unstack(byte_vectors, axis=1)

    cell1 = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell1 for _ in range(2)])
    _, encoding = tf.nn.static_rnn(cell, byte_list, dtype=tf.float32)
    dense1=tf.layers.dense(encoding,MAX_LABEL, activation=None)
    logits = tf.layers.dense(dense1, MAX_LABEL, activation=None)


    return logits, word_list


def data_read_words():
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


def plot_figure(figure, layer_name, layer):
    plt.figure()
    plt.gray()
    plt.subplot(3, 1, 1), plt.axis('off'), plt.imshow(layer[0, :, :, 0])
    plt.subplot(3, 1, 2), plt.axis('off'), plt.imshow(layer[0, :, :, 1])
    plt.subplot(3, 1, 3), plt.axis('off'), plt.imshow(layer[0, :, :, 2])
    plt.savefig('./figures/' + figure + '_' + layer_name + '.png')
    plt.show()

def main():
    global n_words
    trainX, trainY, testX, testY, n_words = data_read_words()

    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)

    logits, word_list = rnn_model(x)

    entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
    train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

    correct_prediction = tf.equal(tf.argmax(logits, 1), y_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    N = len(trainX)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        test_acc = []
        train_err = []
        for i in range(epochs):
            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_op.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})

            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
            train_err.append(entropy.eval(feed_dict={x: trainX, y_: trainY}))
            if i % 10 == 0:

                print('iter %d: test accuracy %g' % (i, test_acc[i]))
                print('iter %d: cross entropy %g' % (i, train_err[i]))
        plot_err_acc(train_err, test_acc)


if __name__ == '__main__':
    main()
