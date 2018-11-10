#
# Project 2, starter code Part a
#

import math
import tensorflow as tf
import numpy as np
import pylab as plt
import pickle
import os
if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')


NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
learning_rate = 0.001
epochs = 401
batch_size = 128

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)


def load_data(file):
    with open(file, 'rb') as fo:
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  #python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding='latin1')

    data, labels = samples['data'], samples['labels']

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    
    labels_ = np.zeros([labels.shape[0], NUM_CLASSES])
    labels_[np.arange(labels.shape[0]), labels-1] = 1

    return data, labels_


def cnn(images):

    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
    
    #Conv 1
    W1 = tf.Variable(tf.truncated_normal([9, 9, NUM_CHANNELS, 50], stddev=1.0/np.sqrt(NUM_CHANNELS*9*9)), name='weights_1')
    b1 = tf.Variable(tf.zeros([50]), name='biases_1')
    conv_1 = tf.nn.relu(tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='VALID') + b1)
    
    #Pool 1
    pool_1 = tf.nn.max_pool(conv_1, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_1')

    
    #Conv 2
    W2 = tf.Variable(tf.truncated_normal([5, 5, 50, 60], stddev=1.0/np.sqrt(NUM_CHANNELS*5*5)), name='weights_2')
    b2 = tf.Variable(tf.zeros([60]), name='biases_2')
    conv_2 = tf.nn.relu(tf.nn.conv2d(pool_1, W2, [1, 1, 1, 1], padding='VALID') + b2)
    
    #Pool 2
    pool_2 = tf.nn.max_pool(conv_2, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_2')
    
    #Fully connected layer    
    dim = pool_2.get_shape()[1].value * pool_2.get_shape()[2].value * pool_2.get_shape()[3].value 
    W_fc = tf.Variable(tf.truncated_normal([dim, 300], stddev=1/np.sqrt(dim)), name='weights_fc')
    b_fc = tf.Variable(tf.zeros([300]), name='biases_fc')
    pool_2_flat = tf.reshape(pool_2, [-1, dim])
    fc = tf.nn.relu(tf.matmul(pool_2_flat, W_fc) + b_fc)
    
    #Softmax
    W3 = tf.Variable(tf.truncated_normal([300, NUM_CLASSES], stddev=1.0/np.sqrt(300)), name='weights_3')
    b3 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases_3')
    logits = tf.matmul(fc, W3) + b3

    return conv_1, pool_1, conv_2, pool_2, logits

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

    trainX, trainY = load_data('data_batch_1')
    print(trainX.shape, trainY.shape)
    
    testX, testY = load_data('test_batch_trim')
    print(testX.shape, testY.shape)

    trainX = (trainX - np.min(trainX, axis = 0))/np.max(trainX, axis = 0)
    testX = (testX - np.min(testX, axis = 0))/np.max(testX, axis = 0)
    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE*NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])
    
    c1, p1, c2, p2, logits = cnn(x)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
        
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    N = len(trainX)
    idx = np.arange(N)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        test_acc = []
        train_err = []
        for i in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]
            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_step.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})

            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
            train_err.append(loss.eval(feed_dict={x: trainX, y_: trainY}))
            if i%100 == 0:
                print('iter %d: test accuracy %g'%(i, test_acc[i]))
                print('iter %d: cross entropy %g'%(i, train_err[i]))
        # Part a: plot training cost and test accuracy
        plot_err_acc(train_err, test_acc)

        # Part b: plot feature maps of two random test patterns
        n = len(testX)
        ind_1 = np.random.randint(low=0, high=n)
        ind_2 = np.random.randint(low=0, high=n)
        X1 = testX[ind_1]
        X2 = testX[ind_2]
        c1_, p1_, c2_, p2_ = sess.run([c1, p1, c2, p2], {x: X1.reshape(-1, 32*32*3)})
        for i in range(50):
            plt.subplot(5, 10, i+1)
            plt.axis('off')
            plt.imshow(c1_[0,:,:,i])
        plt.savefig('./figure1_conv1.png')
        for i in range(50):
            plt.subplot(5, 10, i+1)
            plt.axis('off')
            plt.imshow(p1_[0,:,:,i])
        plt.savefig('./figure1_pool1.png')
        for i in range(60):
            plt.subplot(6, 10, i+1)
            plt.axis('off')
            plt.imshow(c2_[0,:,:,i])
        plt.savefig('./figure1_conv2.png')
        for i in range(60):
            plt.subplot(6, 10, i+1)
            plt.axis('off')
            plt.imshow(p2_[0,:,:,i])
        plt.savefig('./figure1_pool2.png')
        c1_, p1_, c2_, p2_ = sess.run([c1, p1, c2, p2], {x: X2.reshape(-1, 32*32*3)})
        for i in range(50):
            plt.subplot(5, 10, i+1)
            plt.axis('off')
            plt.imshow(c1_[0,:,:,i])
        plt.savefig('./figure2_conv1.png')
        for i in range(50):
            plt.subplot(5, 10, i+1)
            plt.axis('off')
            plt.imshow(p1_[0,:,:,i])
        plt.savefig('./figure2_pool1.png')
        for i in range(60):
            plt.subplot(6, 10, i+1)
            plt.axis('off')
            plt.imshow(c2_[0,:,:,i])
        plt.savefig('./figure2_conv2.png')
        for i in range(60):
            plt.subplot(6, 10, i+1)
            plt.axis('off')
            plt.imshow(p2_[0,:,:,i])
        plt.savefig('./figure2_pool2.png')

if __name__ == '__main__':
    main()
