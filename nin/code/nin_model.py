import tensorflow as tf
import numpy as np
import ipdb
import random
import sys
import os
import logging
import pdb

from gen_utils import *

layers = tf.contrib.layers

class NiNModel():
    def __init__(self, FLAGS):
        # Collect Hyperparameters in config object
        self.config = FLAGS

        self.train_step_count = 0
        self.test_step_count = 0

        # Get global step for training
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        if FLAGS.lr_decay is None:
            self.lr = FLAGS.lr
        else:
            self.lr = tf.train.exponential_decay(self.config.lr, self.global_step, int(self.config.lr_decay_epochs* 50000 / self.config.bsz), self.config.lr_decay, staircase=True) 

        # Initialize model
        self.setup_model()

        # Initialize model saver
        self.saver = tf.train.Saver(tf.global_variables())

    def setup_model(self):
        # Placeholders for input and labels
        self.x = tf.placeholder(tf.float32, shape=[None,32,32,3])
        self.y = tf.placeholder(tf.int32, shape=[None,])

        with tf.name_scope('weights'):
            # first 5x5 > 1x1 > 1x1 pool
            h_conv0 = layers.conv2d(inputs=self.x, num_outputs=192, kernel_size=[5,5], activation_fn=tf.nn.relu)
            h_conv1 = layers.conv2d(inputs=h_conv0, num_outputs=160, kernel_size=[1,1], activation_fn=tf.nn.relu)
            h_conv2 = layers.conv2d(inputs=h_conv1, num_outputs=96, kernel_size=[1,1], activation_fn=tf.nn.relu)
            h_pool2 = layers.max_pool2d(h_conv2, kernel_size=[3,3], stride=2)

            # second 5x5 > 1x1 > 1x1 > pool
            h_conv3 = layers.conv2d(inputs=h_pool2, num_outputs=192, kernel_size=[5,5], activation_fn=tf.nn.relu)
            h_conv4 = layers.conv2d(inputs=h_conv3, num_outputs=192, kernel_size=[1,1], activation_fn=tf.nn.relu)
            h_conv5 = layers.conv2d(inputs=h_conv4, num_outputs=192, kernel_size=[1,1], activation_fn=tf.nn.relu)
            h_pool5 = layers.avg_pool2d(h_conv5, kernel_size=[3,3], stride=2)

            # third 5x5 > 1x1 > 1x1 > pool
            h_conv6 = layers.conv2d(inputs=h_pool5, num_outputs=192, kernel_size=[5,5], activation_fn=tf.nn.relu)
            h_conv7 = layers.conv2d(inputs=h_conv6, num_outputs=192, kernel_size=[1,1], activation_fn=tf.nn.relu)
            h_conv8 = layers.conv2d(inputs=h_conv7, num_outputs=10, kernel_size=[1,1], activation_fn=tf.nn.relu)
            h_pool8 = tf.reshape(layers.avg_pool2d(inputs=h_conv8, kernel_size=[7,7], stride=1), (-1, 10))

        # collect and process all variables
        self.preds = tf.nn.softmax(h_pool8)
        correct_prediction = tf.equal(tf.cast(tf.argmax(self.preds, axis=1), tf.int32), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.end_points = [h_conv0, h_conv1, h_pool2, h_conv3, h_conv4, h_pool5, h_conv6, h_conv7, self.preds, self.accuracy]

        # print Activation shapes
        print('Activation shapes:')
        for activation in self.end_points:
            print(activation.name, ':', activation.get_shape())
        
        self.CE = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=h_pool8))
        tf.summary.scalar('ce_loss', self.CE)
        self.train_step = layers.optimize_loss(loss=self.CE, global_step=self.global_step, learning_rate=self.lr, optimizer=get_optimizer(self.config))
    
    def train(self, sess):
        self.best_acc = 0

        # set up logging
        logging.basicConfig(level=logging.DEBUG)
        if not os.path.exists('./model_output'):
            os.mkdir('model_output')
        file_handler = logging.FileHandler(os.path.join('./model_output', 'log.txt'))
        logging.getLogger().addHandler(file_handler)

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('train',
                                                      sess.graph)
        self.test_writer = tf.summary.FileWriter('test')

        # initialize all variables
        sess.run(tf.global_variables_initializer())

        # get number of params
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        logging.info('Number of params: {}'.format(num_params))

        # write out tensorboard graph
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        writer.close()


        # get dataset 
        all_data = np.load('cifar_data')
        self.train_x = all_data['train_data']
        self.train_y = all_data['train_labels']
        self.test_x  = all_data['test_data']
        self.test_y  = all_data['test_labels']
        
        self.preprocess_data() 

        num_train = self.train_y.size
        order = list(range(num_train))

        # Begin training
        for ep in range(self.config.epochs):
            logging.info('\n *** Epoch {} ***'.format(ep))
            
            losses = []
            accuracies = []

            num_batches = 1 + int(num_train / self.config.bsz)
            prog = Progbar(target=num_batches)
            random.shuffle(order)
            for i in range(num_batches):
                self.train_step_count += 1
                indices = order[i * self.config.bsz: (i+1) * self.config.bsz]

                train_x = self.train_x[indices]
                train_y = self.train_y[indices]

                loss, accuracy = self.optimize(sess, train_x, train_y, i)
                prog.update(i, [('train loss', loss), ('train acc', accuracy)])
                losses.append(loss)
                accuracies.append(accuracy)

            avg_loss = float(sum(losses)) / len(losses)
            avg_acc  = float(sum(accuracies)) / len(accuracies)

            print('')
            logging.info('Train Loss: {}'.format(avg_loss))
            logging.info('Train Acc:  {}'.format(avg_acc))

            val_loss, val_acc = self.evaluate(sess)
            print('')
            logging.info('Valid Loss: {}'.format(val_loss))
            logging.info('Valid Acc:  {}'.format(val_acc))

            if(val_acc > self.best_acc):
                self.best_acc = val_acc
                logging.info('Best accuracy so far!')
                self.saver.save(sess, 'train/model_output', ep)

    def optimize(self, sess, train_x, train_y, i):

        input_feed = {}

        input_feed[self.x] = train_x
        input_feed[self.y] = train_y
        
        if i % 10 == 0:
            summary, _, accuracy, loss = sess.run([self.merged, self.train_step, self.accuracy, self.CE], input_feed)
            self.train_writer.add_summary(summary, self.train_step_count)
        else:
            _, accuracy, loss = sess.run([self.train_step, self.accuracy, self.CE], input_feed)

        #accuracy, loss = sess.run([self.accuracy, self.CE], input_feed)
        
        return loss, accuracy

    def evaluate(self, sess):
        num_test = self.test_y.size
        order = range(num_test)

        valid_bsz = self.config.bsz * 5
        num_batches = int(num_test / valid_bsz) + 1
        val_prog = Progbar(target=num_batches)

        losses = []
        accuracies = []
        
        for i in range(num_batches):
            self.test_step_count += 1
            indices = order[i * valid_bsz: (i+1) * valid_bsz]

            test_x = self.test_x[indices]
            test_y = self.test_y[indices]

            loss, accuracy = self.test(sess, test_x, test_y)
            losses.append(loss)
            accuracies.append(accuracy)
            val_prog.update(i, [('valid loss', loss), ('valid acc', accuracy)])

        avg_loss = float(sum(losses)) / len(losses)
        avg_accuracy = float(sum(accuracies)) / len(accuracies)

        return avg_loss, avg_accuracy

    def test(self, sess, valid_x, valid_y):
        input_feed = {}

        input_feed[self.x] = valid_x
        input_feed[self.y] = valid_y

        summary, accuracy, loss = sess.run([self.merged, self.accuracy, self.CE], input_feed)

        self.test_writer.add_summary(summary, self.test_step_count)
        return loss, accuracy

    def preprocess_data(self):
        MEAN = np.asarray([120.7094049 ,  120.70852412,  120.70476635])
        STD  = np.asarray([64.14942143,  64.1470766 ,  64.15372938])
        self.train_x = (self.train_x - MEAN) / STD
        self.test_x  = (self.test_x  - MEAN) / STD
        
