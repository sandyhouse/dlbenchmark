"""Base model for benchmarks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class Model(object):
    """Base Model for Bechmarks."""

    def __init__(self, 
                 model_name, 
                 batch_size, 
                 learning_rate,
                 fp16_loss_scale, 
                 params=None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.fp16_loss_scale = fp16_loss_scale
        self.params = params
        self.data_type = 'float32'
        self.data_format = params.data_format if params else 'NCHW'
        if params and params.use_fp16:
            self.data_type = 'float16'

    def get_model_name(self):
        return self.model_name

    def get_batch_size(self):
        return self.batch_size

    def get_learning_rate(self):
        return self.learning_rate

    def get_fp16_loss_scale(self):
        return self.fp16_loss_scale

    def get_data_type(self):
        return self.data_type

    def get_data_format(self):
        return self.data_format

    def get_input_data_types(self):
        """Return data types of inputs."""
        raise NotImplementedError('Must be implemented in derived classes')

    def get_input_shapes(self):
        """Return data shapes of inputs."""
        # Each input is of shape [batch_size, height, width, depth]
        # Each label if of shape [batch_size]
        raise NotImplementedError('Must be implemented in derived classes')

class CNNModel(Model):
    """CNN Model for benchmarks."""
    
    def __init__(self, 
                 model_name,
                 batch_size,
                 image_size,
                 learning_rate,
                 layer_counts=None,
                 fp16_loss_scale=128,
                 params=None):
        super(CNNModel, self).__init__(model_name, batch_size, 
                learning_rate, fp16_loss_scale, params)
        self.image_size = image_size
        self.layer_counts = layer_counts
        self.depth = 3

    def get_image_size(self):
        return self.image_size

    def get_layer_counts(self):
        return self.layer_counts

    def get_depth(self):
        return self.depth

    def get_input_shapes(self):
        """Return data shapes of inputs."""
        # Each input is of shape [batch_size, height, width, depth]
        # Each label if of shape [batch_size]
        return [[self.batch_size, self.image_size, self.image_size, self.depth], 
                [self.batch_size]]

    def build_network(self):
        """Build the deep learing network."""
        raise NotImplementedError('Must be implemented in derived classes')

    def train(self, dataset):
        """Train the network."""
        raise NotImplementedError('Must be implemented in derived classes')

    def eval(self, dataset):
        """Evaluate the network."""
        raise NotImplementedError('Must be implemented in derived classes')


    def postprocess(self, results):
        """Postprocess results from model."""
        return results

    def reached_target(self):
        """Define custom functions to stop training"""
        return False

    def get_loss(self, inputs, predictions):
        """Function to get the loss of the model."""
        _, labels = inputs
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=predictions, labels=labels)
        loss = tf.reduce_mean(cross_entropy)
        return loss

    def get_accuracy(self, inputs, predictions):
        """Function to get the accuracy of the model."""
        _, labels = inputs
        top_1 = tf.reduce_sum(
            tf.cast(tf.nn.in_top_k(predictions, labels, 1), self.data_type))
        top_5 = tf.reduce_sum(
            tf.cast(tf.nn.in_top_k(predictions, labels, 5), self.data_type))
        return  {'top_1_accuracy': top_1, 'top_5_accuracy': top_5}
