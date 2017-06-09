# coding: utf-8

import os
import time 

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.examples.tutorials.mnist

N_CLASSES = 10

# Step 1: Read in data
# using TF Learn's built in function to load MNIST data to the folder data/mnist

class MnistModel:
    def __init__(self, batch_size, dropout, n_class, learning_rates):
        self.global_steps = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_steps')
        self.batch_size = batch_size
        self.input_features = 7 * 7 * 64
        self.dropout = dropout
        self.n_class = n_class
        self.learning_rates = learning_rates

    def _create_placeholder(self):
        # Step 3: create placeholders for features and labels
        with tf.name_scope('placeholder'):
            self.images = tf.placeholder(tf.float32, [None, 784], name='images_placeholder')
            self.labels = tf.placeholder(tf.float32, [None, 10], name='labels_placeholder')
            self.dropout = tf.placeholder(tf.float32, name='dropout')

    def _create_conv1_layer(self):
        with tf.variable_scope('conv1') as scope:
            image = tf.reshape(self.images, [-1, 28, 28, 1])
            kernel = tf.get_variable('kernel', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer())
            bias = tf.get_variable('bias', [32], initializer=tf.random_normal_initializer())
            conv_filter = tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME')
            self.conv1 = tf.nn.relu(conv_filter + bias, name=scope.name)
        with tf.variable_scope('pool1') as scope:
            self.pool1 = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1], padding='SAME', name=scope.name)

    def _create_conv2_layer(self):
        with tf.variable_scope('conv2') as scope:
            kernel = tf.get_variable('kernel', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer())
            bias = tf.get_variable('bias', [64], initializer=tf.random_normal_initializer())
            conv_filter = tf.nn.conv2d(self.pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
            self.conv2 = tf.nn.relu(conv_filter + bias, name=scope.name)
        with tf.variable_scope('pool2') as scope:
            self.pool2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1], padding='SAME', name=scope.name)

    def _create_full_connect(self):
        with tf.variable_scope('full_connect') as scope:
            self.pool2 = tf.reshape(self.pool2, [-1, self.input_features])
            w = tf.get_variable('weight', [self.input_features, 1024],
                                initializer=tf.truncated_normal_initializer())
            b = tf.get_variable('bias', [1024], initializer=tf.random_normal_initializer())
            fc = tf.nn.relu(tf.matmul(self.pool2, w) + b, name='relu')
            self.fc = tf.nn.dropout(fc, self.dropout, name='relu_dropout')

    def _create_softmax_loss(self):
        with tf.variable_scope('softmax_linear') as scope:
            w = tf.get_variable('weights', [1024, self.n_class],
                                initializer=tf.truncated_normal_initializer())
            b = tf.get_variable('bias', [self.n_class], initializer=tf.random_normal_initializer())
            self.logits = tf.matmul(self.fc, w) + b

        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            self.loss = tf.reduce_mean(entropy, name='loss')

    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rates).\
            minimize(self.loss, global_step=self.global_steps)

    def _create_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram', self.loss)
            self.summaries = tf.summary.merge_all()

    def create_graphs(self):
        self._create_placeholder()
        self._create_conv1_layer()
        self._create_conv2_layer()
        self._create_full_connect()
        self._create_softmax_loss()
        self._create_optimizer()
        self._create_summaries()


def model_train(model,images, labels, dropout, batch_size, n_epochs, skip_num):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        writer = tf.summary.FileWriter('./my_graph/no_frills/' + str(model.learning_rates), sess.graph)
        total_loss = 0.0
        initial_step = model.global_steps.eval()
        n_batches = int(mnist.train.num_examples/batch_size)
        for index in range(initial_step, initial_step+ n_batches*n_epochs):
            _, loss, summaries = sess.run([model.optimizer, model.loss, model.summaries],
                                          feed_dict={model.images: images,
                                                     model.labels: labels, model.dropout: dropout})
            total_loss += loss
            writer.add_summary(summary=summaries, global_step=index)

            if index%skip_num == 0:
                print('Average loss at step {}: {:5.1f}'.format(index + 1, total_loss/skip_num))
                total_loss = 0.0
                saver.save(sess, r'checkpoints\minist', global_step=index)
        writer.close()
    print("Optimization Finished!")


def model_test(model, batch_size, dropout):
    n_batches = int(mnist.test.num_examples/batch_size)
    total_correct_preds = 0
    with tf.Session() as sess:
        for i in range(n_batches):
            X_batch, Y_batch = mnist.test.next_batch(batch_size)
            _, loss_batch, logits_batch = sess.run([model.optimizer, model.loss, model.logits],
                                                   feed_dict={model.images: X_batch,
                                                              model.labels: Y_batch, model.dropout: dropout})
            preds = tf.nn.softmax(logits=logits_batch)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
            accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
            total_correct_preds += sess.run(accuracy)
    print("Accuracy {0}".format(total_correct_preds / mnist.test.num_examples))

if __name__ == "__main__":

    mnist = input_data.read_data_sets("r'\PycharmProjects\deeplearning\MNIST_data", one_hot=True)

    # Define paramaters for the model
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    SKIP_STEP = 10
    DROPOUT = 0.75
    N_EPOCHS = 50
    N_CLASS = 10

    X_batch, Y_batch = mnist.train.next_batch(BATCH_SIZE)
    model = MnistModel(BATCH_SIZE, DROPOUT, N_CLASS, LEARNING_RATE)
    model.create_graphs()
    model_train(model, X_batch, Y_batch, DROPOUT, BATCH_SIZE, N_EPOCHS, SKIP_STEP)
    model_test(model, BATCH_SIZE, DROPOUT)

