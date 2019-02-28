"""
@file    Network.py
@author  rohithjayarajan
@date 02/22/2019

Licensed under the
GNU General Public License v3.0
"""

import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True


class DeepNetwork:
    # def __init__(self):
        # self.ImageSize = ImageSize
        # self.MiniBatchSize = MiniBatchSize

    def HomographyNet(self, Img, isTrain):
        """
        Inputs: 
        Img is a MiniBatch of the current image
        ImageSize - Size of the Image
        Outputs:
        prLogits - logits output of the network
        prSoftMax - softmax output of the network
        """

        #############################
        # Fill your network here!
        #############################
        with tf.variable_scope("ConvolutionalBlock1"):
            conv1 = tf.layers.conv2d(
                inputs=Img,
                filters=64,
                kernel_size=[3, 3],
                padding="same",
                name='conv1')
            bn1 = tf.layers.batch_normalization(conv1)
            z1 = tf.nn.relu(bn1, name='ReLU1')

        with tf.variable_scope("ConvolutionalBlock2"):
            conv2 = tf.layers.conv2d(
                inputs=z1,
                filters=64,
                kernel_size=[3, 3],
                padding="same",
                name='conv2')
            bn2 = tf.layers.batch_normalization(conv2)
            z2 = tf.nn.relu(bn2, name='ReLU2')

        pool1 = tf.layers.max_pooling2d(
            z2,
            pool_size=[2, 2],
            strides=2,
            padding='valid',
            name='pool1')

        with tf.variable_scope("ConvolutionalBlock3"):
            conv3 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[3, 3],
                padding="same",
                name='conv3')
            bn3 = tf.layers.batch_normalization(conv3)
            z3 = tf.nn.relu(bn3, name='ReLU3')

        with tf.variable_scope("ConvolutionalBlock4"):
            conv4 = tf.layers.conv2d(
                inputs=z3,
                filters=64,
                kernel_size=[3, 3],
                padding="same",
                name='conv4')
            bn4 = tf.layers.batch_normalization(conv4)
            z4 = tf.nn.relu(bn4, name='ReLU4')

        pool2 = tf.layers.max_pooling2d(
            z4,
            pool_size=[2, 2],
            strides=2,
            padding='valid',
            name='pool2')

        with tf.variable_scope("ConvolutionalBlock5"):
            conv5 = tf.layers.conv2d(
                inputs=pool2,
                filters=128,
                kernel_size=[3, 3],
                padding="same",
                name='conv5')
            bn5 = tf.layers.batch_normalization(conv5)
            z5 = tf.nn.relu(bn5, name='ReLU5')

        with tf.variable_scope("ConvolutionalBlock6"):
            conv6 = tf.layers.conv2d(
                inputs=z5,
                filters=128,
                kernel_size=[3, 3],
                padding="same",
                name='conv6')
            bn6 = tf.layers.batch_normalization(conv6)
            z6 = tf.nn.relu(bn6, name='ReLU6')

        pool3 = tf.layers.max_pooling2d(
            z6,
            pool_size=[2, 2],
            strides=2,
            padding='valid',
            name='pool3')

        with tf.variable_scope("ConvolutionalBlock7"):
            conv7 = tf.layers.conv2d(
                inputs=pool3,
                filters=128,
                kernel_size=[3, 3],
                padding="same",
                name='conv7')
            bn7 = tf.layers.batch_normalization(conv7)
            z7 = tf.nn.relu(bn7, name='ReLU7')

        with tf.variable_scope("ConvolutionalBlock8"):
            conv8 = tf.layers.conv2d(
                inputs=z7,
                filters=128,
                kernel_size=[3, 3],
                padding="same",
                name='conv8')
            bn8 = tf.layers.batch_normalization(conv8)
            z8 = tf.nn.relu(bn8, name='ReLU8')

        dropout1 = tf.layers.dropout(inputs=z8, rate=0.5, training=isTrain)

        z8_flat = tf.layers.flatten(dropout1)
        dense1 = tf.layers.dense(inputs=z8_flat, units=1024, activation=None)
        bn9 = tf.layers.batch_normalization(dense1)
        z9 = tf.nn.relu(bn9, name='ReLU9')
        dropout2 = tf.layers.dropout(inputs=z9, rate=0.5, training=isTrain)

        z10 = tf.layers.dense(inputs=dropout2, units=8, activation=None)
        H4Pt = z10

        return H4Pt
