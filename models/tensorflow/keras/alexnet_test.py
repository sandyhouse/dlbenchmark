"""Test Program for AlexNet Model."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10

from models.tensorflow.keras import alexnet

if __name__ == '__main__':
    alex = alexnet.AlexnetModel()
    (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
    print(len(train_data))
    print(len(test_data))
