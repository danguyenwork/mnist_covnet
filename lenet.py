from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten

def _pad_for_lenet(data):
    height_pad = int(max(0, 32 - data.shape[1]) / 2)
    width_pad = int(max(0, 32 - data.shape[2]) / 2)
    return np.pad(data, ((0,0),(height_pad,height_pad),(width_pad,width_pad),(0,0)), 'constant')

def _extract_data():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True, reshape = False)

    x_train = mnist.train.images
    y_train = mnist.train.labels

    x_train, y_train = shuffle(x_train, y_train)

    x_valid = mnist.validation.images
    y_valid = mnist.validation.labels

    x_test = mnist.test.images
    y_test = mnist.test.labels

    x_train = _pad_for_lenet(x_train)
    x_valid = _pad_for_lenet(x_valid)
    x_test = _pad_for_lenet(x_test)

    return mnist, x_train, y_train, x_valid, y_valid, x_test, y_test

def LeNet(x):
    # con layer1
    conv1_w = tf.Variable(tf.truncated_normal([5,5,1,6]))
    conv1_b = tf.Variable(tf.zeros([6]))
    conv1 = tf.nn.conv2d(x, conv1_w, strides = [1,1,1,1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1],padding = 'VALID')

    # con layer2
    conv2_w = tf.Variable(tf.truncated_normal([5,5,6,16]))
    conv2_b = tf.Variable(tf.zeros([16]))
    conv2 = tf.nn.conv2d(conv1, conv2_w, strides = [1,1,1,1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1],padding = 'VALID')

    fc0 = flatten(conv2)

    # fc layer 1
    fc1_w = tf.Variable(tf.truncated_normal([400,120]))
    fc1_b = tf.Variable(tf.zeros([120]))
    fc1 = tf.matmul(fc0, fc1_w) + fc1_b
    fc1 = tf.nn.relu(fc1)

    # fc layer 2
    fc2_w = tf.Variable(tf.truncated_normal([120,84]))
    fc2_b = tf.Variable(tf.zeros([84]))
    fc2 = tf.matmul(fc1, fc2_w) + fc2_b
    fc2 = tf.nn.relu(fc2)

    # fc layer 3
    fc3_w  = tf.Variable(tf.truncated_normal([84,10]))
    fc3_b  = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_w) + fc3_b

    return logits


IMG_SHAPE = 32
EPOCHS = 10
BATCH_SIZE = 128
N_CLASSES = 10

mnist, x_train, y_train, x_valid, y_valid, x_test, y_test = _extract_data()

x = tf.placeholder(tf.float32, [None, IMG_SHAPE, IMG_SHAPE, 1])
y = tf.placeholder(tf.float32, [None, N_CLASSES])

rate = 0.001

logits = LeNet(x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate = rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(x_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        x_train, y_train = shuffle(x_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(x_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Model saved")
