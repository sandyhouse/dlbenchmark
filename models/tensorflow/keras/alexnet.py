# -*- coding: utf-8 -*-
"""Alexnet model with Keras API.
References:
    Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton.
    ImageNet Classification with Deep Convolutional Neural Networks.
    In proceedings of the 25th International Conference on Neural
    Information Processing Systems - Volume 1 (NIPS'12).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from models.tensorflow import utils

#from models import model

#class AlexnetModel(model.CNNModel):
class AlexnetModel():
    """Alexnet with Keras API."""

    def __init__(self, flags_obj):
        #super(AlexnetModel, self).__init__(
        #    'alexnet', 512, 224, 0.005, params=params)
        #self.image_height = 224
        #self.image_width = 224
        self.image_height = 32
        self.image_width = 32
        self.image_depth = 3
        self.num_classes = 1000
        if flags_obj.data_type == 'float32':
            self.data_type = tf.float32
        else:
            raise ValueError('tf.float16 is not supported in keras. '
                    'Use tf.float32 instead.')

    def build_network(self, flags_obj):
        """ Building the AlexNet"""
        num_gpus = flags_obj.num_gpus
        distribution = utils.get_distribution_strategy(num_gpus)
        with distribution.scope():
            model = keras.Sequential()
            # 1st conv layer
            model.add(layers.Conv2D(
                input_shape=(self.image_height, 
                    self.image_width, self.image_depth),
                #filters=96, kernel_size=(11,11),
                #strides=(4,4), padding='same',
                filters=96, kernel_size=(3,3),
                strides=(2,2), padding='same',
                activation='relu'))
            # max pooling
            model.add(layers.MaxPooling2D(
                pool_size=(2,2), strides=(2,2)))
            # batch normalization
            model.add(layers.BatchNormalization())

            # 2nd conv layer
            model.add(layers.Conv2D(
                filters=256, kernel_size=(5,5),
                strides=(1,1), padding='same',
                activation='relu'))
            # max pooling
            model.add(layers.MaxPooling2D(
                pool_size=(3,3), strides=(2,2)))
            # batch normalization
            model.add(layers.BatchNormalization())

            # 3rd conv layer
            model.add(layers.Conv2D(
                filters=384, kernel_size=(3,3),
                strides=(1,1), padding='same',
                activation='relu'))

            # 4th conv layer
            model.add(layers.Conv2D(
                filters=384, kernel_size=(3,3),
                strides=(1,1), padding='same',
                activation='relu'))

            # 5th conv layer
            model.add(keras.layers.Conv2D(
                filters=256, kernel_size=(3,3),
                strides=(1,1), padding='same',
                activation='relu'))
            # max pooling
            model.add(layers.MaxPooling2D(
                pool_size=(3,3), strides=(2,2)))
            # batch normalization
            model.add(layers.BatchNormalization())

            # to a dense layer
            model.add(layers.Flatten())
            # 1st Dense layer
            model.add(layers.Dense(4096,
                activation='tanh'))
            # dropout
            model.add(layers.Dropout(0.5))

            # 2nd Dense layer
            model.add(layers.Dense(4096,
                activation='tanh'))
            # dropout
            model.add(layers.Dropout(0.5))

            # 3rd Dense layer
            model.add(layers.Dense(self.num_classes,
                activation='softmax'))

            model.summary()

            # compile
            model.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])
            self.model = model
    
    def train(self, flags):
        #data_dir = os.path.join(flags.data_dir, 'cifar-10-batches-bin')
        #assert os.path.exists(data_dir), (
        #    "Please make sure 'cifar-10-batch-bin' in " + flags.data_dir + ".")
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        y_train = keras.utils.to_categorical(y_train, 1000)
        print(x_train.shape)
        num_gpus = flags.num_gpus
        distribution = utils.get_distribution_strategy(num_gpus)
        with distribution.scope():
            self.model.fit(x_train, y_train, batch_size=128, epochs = 1000,
                    verbose=1)

    def eval(self,  flags):
        model.evaluate(data, labels, self.batch_size)

