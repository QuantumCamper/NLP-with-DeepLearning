#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1: A window into NER
"""
import argparse
import sys
import time
import logging
from datetime import datetime

import tensorflow as tf
import numpy as np

from util import print_sentence, write_conll
from data_util import load_and_preprocess_data, load_embeddings, read_conll, ModelHelper
from ner_model import NERModel
from defs import LBLS


logger = logging.getLogger("hw3.q1")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.

    TODO: Fill in what n_window_features should be, using n_word_features and window_size.
    """
    n_word_features = 2 # Number of features for every word in the input.
    window_size = 1 # The size of the window to use.
    n_window_features = (2*window_size + 1)* n_word_features# The total number of features used for each window.
    n_classes = 5
    dropout = 0.5
    embed_size = 50
    hidden_size = 200
    batch_size = 2048
    n_epochs = 10
    lr = 0.001

    def __init__(self, output_path=None):
        if output_path:
            # Where to save things.
            self.output_path = output_path
        else:
            self.output_path = "results/window/{:%Y%m%d_%H%M%S}/".format(datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.log_output = self.output_path + "log"
        self.conll_output = self.output_path + "window_predictions.conll"


def make_windowed_data(data, start, end, window_size = 1):
    """Uses the input sequences in @data to construct new windowed data points.
    """
    windowed_data = []
    for sentence, labels in data:
        for index in range(len(sentence)):
            if len(sentence) == 1:
                sent = sentence[0]
                window_sentence = start * (window_size - index) + sent + end*window_size
                window_labels = labels[index]
            elif index <= window_size - 1:
                sent = [sub_item for item in sentence[index: index + window_size + 1] for sub_item in item]
                window_sentence = start*(window_size - index) + sent
                window_labels = labels[index]
            elif index == len(sentence) - window_size:
                sent = [sub_item for item in sentence[index - window_size:] for sub_item in item]
                window_sentence = sent + end*window_size
                window_labels = labels[index]
            else:
                sent = [sub_item for item in sentence[index - window_size: (index + window_size + 1)]
                        for sub_item in item]
                window_sentence = sent
                window_labels = labels[index]
            windowed_data.append((window_sentence, window_labels))

    return windowed_data

class WindowModel(NERModel):
    """
    Implements a feedforward neural network with an embedding layer and
    single hidden layer.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        """
        with tf.name_scope('placeholder'):
            self.input_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, self.config.n_window_features])
            self.labels_placeholder = tf.placeholder(dtype=tf.int32, shape=[None])
            self.dropout_placeholder = tf.placeholder(dtype=tf.float32, shape=[])

    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1):
        """Creates the feed_dict for the model.
        """
        if labels_batch is None:

            feed_dict = {self.input_placeholder: inputs_batch,
                         self.dropout_placeholder: dropout}
        else:
            feed_dict = {self.input_placeholder: inputs_batch,
                         self.labels_placeholder: labels_batch,
                         self.dropout_placeholder: dropout}


        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:
        """
        with tf.name_scope('embedding'):
            embed_tensor = tf.Variable(self.pretrained_embeddings, trainable=False, name='embedding_tensor')
            embeddings = tf.nn.embedding_lookup(embed_tensor, self.input_placeholder)
            embeddings = tf.reshape(embeddings, shape=[-1, self.config.n_window_features*self.config.embed_size])

        return embeddings

    def add_prediction_op(self):
        """Adds the 1-hidden-layer NN:
            h = Relu(xW + b1)
            h_drop = Dropout(h, dropout_rate)
            pred = h_dropU + b2

        """
        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder
        with tf.variable_scope('predict', initializer=tf.contrib.layers.xavier_initializer()) as vs:
            W = tf.get_variable('W', dtype=tf.float32, shape=[self.config.n_window_features*self.config.embed_size,
                                                              self.config.hidden_size])
            b1 = tf.get_variable('b1', dtype=tf.float32, shape=[self.config.hidden_size])
            h = tf.nn.relu_layer(x=x, weights=W, biases=b1)
            h_drop = tf.nn.dropout(h, keep_prob=dropout_rate)
            U = tf.get_variable('U', dtype=tf.float32, shape=[self.config.hidden_size, self.config.n_classes])
            b2 = tf.get_variable('b2', dtype=tf.float32, shape=[self.config.n_classes])
            self.pred = tf.matmul(h_drop, U) + b2


    def add_loss_op(self):
        """Adds Ops for the loss function to the computational graph.
        """
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred,
                                                                             labels=self.labels_placeholder))

    def add_training_op(self, loss):
        """Sets up the training Ops.
        """
        self.train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss=self.loss)

    def preprocess_sequence_data(self, examples):
        return make_windowed_data(examples, start=self.helper.START, end=self.helper.END,
                                  window_size=self.config.window_size)

    def consolidate_predictions(self, examples_raw, examples, preds):
        """Batch the predictions into groups of sentence length.
        """
        ret = []
        #pdb.set_trace()
        i = 0
        for sentence, labels in examples_raw:
            labels_ = preds[i:i+len(sentence)]
            i += len(sentence)
            ret.append([sentence, labels, labels_])
        return ret

    def predict_on_batch(self, sess, inputs_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(tf.argmax(self.pred, axis=1), feed_dict=feed)
        return predictions

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch,
                                     dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def build(self):
        self.add_placeholders()
        self.add_embedding()
        self.add_prediction_op()
        self.add_loss_op()
        self.add_training_op(self.loss)

    def __init__(self, helper, config, pretrained_embeddings, report=None):
        super(WindowModel, self).__init__(helper, config, report)
        self.pretrained_embeddings = pretrained_embeddings
        self.config = config

        # Defining placeholders.
        self.input_placeholder = None
        self.labels_placeholder = None
        self.dropout_placeholder = None

        self.build()


def make_windowed_check():
    sentences = [[[1,1], [2,0], [3,3]]]
    sentence_labels = [[1, 2, 3]]
    data = zip(sentences, sentence_labels)
    w_data = make_windowed_data(data, start=[5,0], end=[6,0], window_size=1)

    assert len(w_data) == sum(len(sentence) for sentence in sentences)

    assert w_data == [
        ([5,0] + [1,1] + [2,0], 1,),
        ([1,1] + [2,0] + [3,3], 2,),
        ([2,0] + [3,3] + [6,0], 3,),
        ]

def do_check1(_):
    logger.info("Testing make_windowed_data")
    make_windowed_check()
    logger.info("Passed!")

def do_check2(args):
    logger.info("Testing implementation of WindowModel")
    config = Config()
    helper, train, dev, train_raw, dev_raw = load_and_preprocess_data(args)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = WindowModel(helper, config, embeddings)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = None

        with tf.Session() as session:
            session.run(init)
            model.fit(session, saver, train, dev)

    logger.info("Model did not crash!")
    logger.info("Passed!")

def do_train(args):
    # Set up some parameters.
    config = Config()
    helper, train, dev, train_raw, dev_raw = load_and_preprocess_data(args)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]
    helper.save(config.output_path)

    handler = logging.FileHandler(config.log_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    report = None #Report(Config.eval_output)

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = WindowModel(helper, config, embeddings)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            model.fit(session, saver, train, dev)
            if report:
                report.log_output(model.output(session, dev_raw))
                report.save()
            else:
                # Save predictions in a text file.
                output = model.output(session, dev_raw)
                sentences, labels, predictions = zip(*output)
                predictions = [[LBLS[l] for l in preds] for preds in predictions]
                output = zip(sentences, labels, predictions)

                with open(model.config.conll_output, 'w') as f:
                    write_conll(f, output)
                with open(model.config.eval_output, 'w') as f:
                    for sentence, labels, predictions in output:
                        print_sentence(f, sentence, labels, predictions)

def do_evaluate(args):
    config = Config(args.model_path)
    helper = ModelHelper.load(args.model_path)
    input_data = read_conll(args.data)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = WindowModel(helper, config, embeddings)

        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            saver.restore(session, model.config.model_output)
            for sentence, labels, predictions in model.output(session, input_data):
                predictions = [LBLS[l] for l in predictions]
                print_sentence(args.output, sentence, labels, predictions)

def do_shell(args):
    config = Config(args.model_path)
    helper = ModelHelper.load(args.model_path)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = WindowModel(helper, config, embeddings)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            saver.restore(session, model.config.model_output)

            print("""Welcome!
You can use this shell to explore the behavior of your model.
Please enter sentences with spaces between tokens, e.g.,
input> Germany 's representative to the European Union 's veterinary committee .
""")
            while True:
                # Create simple REPL
                try:
                    sentence = raw_input("input> ")
                    tokens = sentence.strip().split(" ")
                    for sentence, _, predictions in model.output(session, [(tokens, ["O"] * len(tokens))]):
                        predictions = [LBLS[l] for l in predictions]
                        print_sentence(sys.stdout, sentence, [""] * len(tokens), predictions)
                except EOFError:
                    print("Closing session.")
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains and tests an NER model')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('test1', help='')
    command_parser.set_defaults(func=do_check1)

    command_parser = subparsers.add_parser('test2', help='')
    command_parser.add_argument('--data_train', '--data_train', type=argparse.FileType('r'), default="data/tiny.conll",
                                help="Training data")
    command_parser.add_argument('--dd', '--data-dev', type=argparse.FileType('r'), default="data/tiny.conll",
                                help="Dev data")
    command_parser.add_argument('--v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt",
                                help="Path to vocabulary file")
    command_parser.add_argument('--vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt",
                                help="Path to word vectors file")
    command_parser.set_defaults(func=do_check2)

    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-dt', '--data-train', type=argparse.FileType('r'), default="data/train.conll",
                                help="Training data")
    command_parser.add_argument('-dd', '--data-dev', type=argparse.FileType('r'), default="data/dev.conll",
                                help="Dev data")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt",
                                help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt",
                                help="Path to word vectors file")
    command_parser.set_defaults(func=do_train)

    command_parser = subparsers.add_parser('evaluate', help='')
    command_parser.add_argument('-d', '--data', type=argparse.FileType('r'), default="data/dev.conll",
                                help="Training data")
    command_parser.add_argument('-m', '--model-path', help="Training data")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt",
                                help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt",
                                help="Path to word vectors file")
    command_parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout, help="Training data")
    command_parser.set_defaults(func=do_evaluate)

    command_parser = subparsers.add_parser('shell', help='')
    command_parser.add_argument('-m', '--model-path', help="Training data")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt",
                                help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt",
                                help="Path to word vectors file")
    command_parser.set_defaults(func=do_shell)

    ARGS = parser.parse_args(['evaluate'])
    ARGS.func(ARGS)