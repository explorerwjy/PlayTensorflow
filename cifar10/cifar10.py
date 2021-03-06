# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import cifar10_input

FLAGS = tf.app.flags.FLAGS


MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op
IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]

WEIGHT_DECAY = 1
WEIGHT_DECAY_2 = 1
Keep_Prop = 0.5

Keep_Prop=0.5

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 64,
        """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', './cifar10_data',
        """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
        """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 1e-4       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
            tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(
                name, shape, initializer=initializer, dtype=dtype)
        return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
            name,
            shape,
            tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def distorted_inputs():
    """Construct distorted input for CIFAR training using the Reader ops.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.

    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
            batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def inputs(eval_data):
    """Construct input for CIFAR evaluation using the Reader ops.

    Args:
      eval_data: bool, indicating if one should use the train or eval data set.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.

    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.inputs(eval_data=eval_data,
            data_dir=data_dir,
            batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def inference_0(images):
    """Build the CIFAR-10 model.

    Args:
      images: Images returned from distorted_inputs() or inputs().

    Returns:
      Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                shape=[5, 5, 3, 64],
                stddev=5e-2,
                wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
            padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
            name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                shape=[5, 5, 64, 64],
                stddev=5e-2,
                wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
            name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                stddev=0.04, wd=0.004)
        biases = _variable_on_cpu(
                'biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) +
                biases, name=scope.name)
        _activation_summary(local3)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                stddev=0.04, wd=0.004)
        biases = _variable_on_cpu(
                'biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) +
                biases, name=scope.name)
        _activation_summary(local4)

    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                stddev=1 / 192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                tf.constant_initializer(0.0))
        softmax_linear = tf.add(
                tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


def inference_1(RawTensor):
    #print RawTensor
    #InputTensor = tf.reshape(RawTensor, [-1, WIDTH, HEIGHT, 3])
    # print InputTensor
    InputTensor = RawTensor
    # ==========================================================================================
    # conv1 3-64
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 3, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(
                InputTensor, kernel, [
                    1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu(
                        'biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)
        print conv1
    # ==========================================================================================
    # ==========================================================================================
    # MaxPooling
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[
        1, 2, 2, 1], padding='SAME', name='pool1')
    # ==========================================================================================
    # ==========================================================================================
    # conv3 3-128
    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 64, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(
                pool1, kernel, [
                    1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu(
                        'biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv3)
        print conv3
    # ==========================================================================================
    # ==========================================================================================
    # MaxPooling
    pool2 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[
        1, 2, 2, 1], padding='SAME', name='pool2')
    # ==========================================================================================
    # ==========================================================================================
    # conv5 3-256
    with tf.variable_scope('conv5') as scope:
        kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 64, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(
                pool2, kernel, [
                    1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu(
                        'biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv5)
        print conv5
    # ==========================================================================================
    # ==========================================================================================
    # conv6 3-256
    with tf.variable_scope('conv6') as scope:
        kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(
                conv5, kernel, [
                    1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu(
                        'biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv6 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv6)
        print conv6
    # ==========================================================================================
    # ==========================================================================================
    # MaxPooling
    pool3 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[
        1, 2, 2, 1], padding='SAME', name='pool3')
    # ==========================================================================================
    # ==========================================================================================
    # conv9 3-512
    with tf.variable_scope('conv9') as scope:
        kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 128, 256], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(
                pool3, kernel, [
                    1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu(
                        'biases', [256], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv9 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv9)
        print conv9
    # ==========================================================================================
    # ==========================================================================================
    # conv10 3-512
    with tf.variable_scope('conv10') as scope:
        kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 256, 256], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(
                conv9, kernel, [
                    1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu(
                        'biases', [256], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv10 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv10)
        print conv10
    # ==========================================================================================
    # ==========================================================================================
    # MaxPooling
    pool4 = tf.nn.max_pool(conv10, ksize=[1, 2, 2, 1], strides=[
        1, 2, 2, 1], padding='SAME', name='pool4')
    # ==========================================================================================
    # ==========================================================================================
    # conv13 3-512
    with tf.variable_scope('conv13') as scope:
        kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 256, 512], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(
                pool4, kernel, [
                    1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu(
                        'biases', [512], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv13 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv13)
        print conv13
    # ==========================================================================================
    # ==========================================================================================
    # conv14 3-512
    with tf.variable_scope('conv14') as scope:
        kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 512, 512], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(
                conv13, kernel, [
                    1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu(
                        'biases', [512], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv14 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv14)
        print conv14
    # ==========================================================================================
    # ==========================================================================================
    # MaxPooling
    pool5 = tf.nn.max_pool(conv14, ksize=[1, 2, 2, 1], strides=[
        1, 2, 2, 1], padding='SAME', name='pool5')
    # ==========================================================================================
    # ==========================================================================================
    # local1
    with tf.variable_scope('local1') as scope:
        reshape = tf.reshape(pool5, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay(
                'weights', shape=[dim, 4096], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu(
                'biases', [4096], tf.constant_initializer(0.1))
        local1 = tf.nn.relu(
                tf.matmul(
                    reshape,
                    weights) + biases,
                name=scope.name)
        #_activation_summary(local1)
        local1_drop = tf.nn.dropout(local1, Keep_Prop)
        _activation_summary(local1_drop)
    print local1
    # ==========================================================================================
    # ==========================================================================================
    # local2
    with tf.variable_scope('local2') as scope:
        weights = _variable_with_weight_decay(
                'weights', shape=[4096, 4096], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu(
                'biases', [4096], tf.constant_initializer(0.1))
        local2 = tf.nn.relu(
                tf.matmul(
                    local1_drop,
                    weights) + biases,
                name=scope.name)
        local2_drop = tf.nn.dropout(local2, Keep_Prop)
        _activation_summary(local2_drop)
    print local2
    # ==========================================================================================
    # ==========================================================================================
    # local3
    with tf.variable_scope('local3') as scope:
        weights = _variable_with_weight_decay(
                'weights', shape=[4096, 1000], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu(
                'biases', [1000], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(
                tf.matmul(
                    local2_drop,
                    weights) + biases,
                name=scope.name)
        #local3_drop = tf.nn.dropout(local3, 0.5)
        _activation_summary(local3)
    print local3
    # ==========================================================================================
    # ==========================================================================================
    # linear layer (WX + b)
    with tf.variable_scope('softmax') as scope:
        weights = _variable_with_weight_decay(
                'weights', [1000, NUM_CLASSES], stddev=1 / 1000.0, wd=0.0)
        biases = _variable_on_cpu(
                'biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        softmax_linear = tf.add(
                tf.matmul(
                    local3,
                    weights),
                biases,
                name=scope.name)
        _activation_summary(softmax_linear)
        #softmax = tf.nn.softmax(softmax_linear, dim=-1, name=None)
    print softmax_linear
    # ==========================================================================================
    return softmax_linear

class Config:
    def __init__(self):
        root = self.Scope('')
        for k, v in FLAGS.__dict__['__flags'].iteritems():
            root[k] = v
        self.stack = [ root ]

    def iteritems(self):
        return self.to_dict().iteritems()

    def to_dict(self):
        self._pop_stale()
        out = {}
        # Work backwards from the flags to top fo the stack
        # overwriting keys that were found earlier.
        for i in range(len(self.stack)):
            cs = self.stack[-i]
            for name in cs:
                out[name] = cs[name]
        return out

    def _pop_stale(self):
        var_scope_name = tf.get_variable_scope().name
        top = self.stack[0]
        while not top.contains(var_scope_name):
            # We aren't in this scope anymore
            self.stack.pop(0)
            top = self.stack[0]

    def __getitem__(self, name):
        self._pop_stale()
        # Recursively extract value
        for i in range(len(self.stack)):
            cs = self.stack[i]
            if name in cs:
                return cs[name]

        raise KeyError(name)

    def set_default(self, name, value):
        if not name in self:
            self[name] = value

    def __contains__(self, name):
        self._pop_stale()
        for i in range(len(self.stack)):
            cs = self.stack[i]
            if name in cs:
                return True
        return False

    def __setitem__(self, name, value):
        self._pop_stale()
        top = self.stack[0]
        var_scope_name = tf.get_variable_scope().name
        assert top.contains(var_scope_name)

        if top.name != var_scope_name:
            top = self.Scope(var_scope_name)
            self.stack.insert(0, top)

        top[name] = value

    class Scope(dict):
        def __init__(self, name):
            self.name = name

        def contains(self, var_scope_name):
            return var_scope_name.startswith(self.name)

class ResNet():
    def __init__(self):
        self.activation = tf.nn.relu

    def Inference(self, x, is_training=True, num_class=3, num_blocks=[3, 4, 6, 3], use_bias=False, bottleneck=True):
        c = Config()
        c["bottleneck"] = bottleneck
        c["is_training"] = tf.convert_to_tensor(is_training, dtype="bool", name="is_training")
        c["ksize"] = 3
        c["stride"] = 1
        c["use_bias"] = use_bias
        c["fc_units_out"] = num_class
        c["num_blocks"] = num_blocks
        c["stack_stride"] = 2
        print x
        with tf.variable_scope("scale1"):
            c["conv_filters_out"] = 64
            c["ksize"] = 7
            c["stride"] = 2
            x = self.conv(x, c)
            x = self.bn(x, c)
            x = self.activation(x)
        print x
        with tf.variable_scope("scale2"):
            x = self._max_pool(x, ksize=3, stride=2)
            c["num_blocks"] = num_blocks[0]
            c["stack_stride"] = 1
            c["block_filters_internal"] = 64
            x = self.stack(x, c)

        with tf.variable_scope("scale3"):
            #x = _max_pool(x, ksize=3, stride=2)
            c["num_blocks"] = num_blocks[1]
            #c["stack_stride"] = 1
            c["block_filters_internal"] = 128
            assert c["stack_stride"] == 2
            x = self.stack(x, c)

        with tf.variable_scope("scale4"):
            #x = _max_pool(x, ksize=3, stride=2)
            c["num_blocks"] = num_blocks[2]
            #c["stack_stride"] = 1
            c["block_filters_internal"] = 256
            x = self.stack(x, c)

        with tf.variable_scope("scale5"):
            #x = _max_pool(x, ksize=3, stride=2)
            c["num_blocks"] = num_blocks[3]
            #c["stack_stride"] = 1
            c["block_filters_internal"] = 512
            x = self.stack(x, c)

        x = tf.reduce_mean(x, reduction_indices=[1,2], name="avg_pool")

        if num_class != None:
            with tf.variable_scope("fc"):
                x = self.fc(x, c)

        return x

    def stack(self, x, c):
        for n in range(c["num_blocks"]):
            s = c["stack_stride"] if n == 0 else 1
            c["block_stride"] = s
            with tf.variable_scope("block%d"%(n+1)):
                x = self.block(x, c)
            return x

    def block(self, x, c):
        filters_in = x.get_shape()[-1]
        m = 4 if c["bottleneck"] else 1
        filters_out = m * c["block_filters_internal"]
        shortcut = x # branch 1
        c["conv_filters_out"] = c["block_filters_internal"]
        if c["bottleneck"]:
            with tf.variable_scope("a"):
                c["ksize"] = 1
                c["stride"] = c["block_stride"]
                x = self.conv(x, c)
                x = self.bn(x, c)
                x = self.activation(x)
            with tf.variable_scope("b"):
                x = self.conv(x, c)
                x = self.bn(x, c)
                x = self.activation(x)
            with tf.variable_scope("c"):
                c["conv_filters_out"] = filters_out
                c["ksize"] = 1
                assert c["stride"] == 1
                x = self.conv(x, c)
                x = self.bn(x, c)

        else:
            with tf.variable_scope("A"):
                c["stride"] = c["block_stride"]
                assert c["ksize"] == 3
                x = self.conv(x, c)
                x = self.bn(x, c)
                x = self.activation(x)
            with tf.variable_scope("B"):
                c["conv_filters_out"] = filters_out
                assert c["ksize"] == 3
                assert c["stride"] == 1
                x = self.conv(x, c)
                x = self.bn(x, c)

        with tf.variable_scope("shortcut"):
            if filters_out != filters_in or c["block_stride"] != 1:
                c["ksize"] = 1
                c["stride"] = c["block_stride"]
                c["conv_filters_out"] = filters_out
                shortcut = self.conv(shortcut, c)
                shortcut = self.bn(shortcut, c)

            return self.activation(x + shortcut)

    def conv(self, x, c):
        ksize = c["ksize"]
        stride = c["stride"]
        filters_out = c["conv_filters_out"]
        filters_in = x.get_shape()[-1] #Depth
        shape = [ksize, ksize, filters_in, filters_out]
        initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
        weights = self._get_variable("weights", shape=shape, dtype="float", initializer=initializer, weight_decay=CONV_WEIGHT_DECAY)
        return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')

    def bn(self, x, c):
        x_shape = x.get_shape()
        params_shape = x_shape[-1:]
        if c["use_bias"]:
            bias = self._get_variable("bias", params_shape, initializer=tf.zeros_initializer)
            return x + bias
        axis = list(range(len(x_shape) - 1))
        beta = self._get_variable("beta", params_shape, initializer=tf.zeros_initializer)
        gamma = self._get_variable("gamma", params_shape, initializer=tf.zeros_initializer)
        moving_mean = self._get_variable("moving_mean", params_shape, initializer=tf.zeros_initializer, trainable=False)
        moving_variance = self._get_variable("moving_variance", params_shape, initializer=tf.zeros_initializer, trainable=False)
        mean, variance = tf.nn.moments(x, axis)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
        update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
        mean, variance = control_flow_ops.cond(c["is_training"], lambda: (mean, variance), lambda: (moving_mean, moving_variance))
        x =tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
        return x 

    def _max_pool(self, x, ksize=3, stride=2):
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding="SAME")

    def fc(self, x, c):
        num_units_in = x.get_shape()[-1]
        num_units_out = c["fc_units_out"]
        weights_initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV)
        weights = self._get_variable("weights", shape=[num_units_in, num_units_out], initializer=weights_initializer, weight_decay=FC_WEIGHT_DECAY)
        biases = self._get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer)
        x = tf.nn.xw_plus_b(x, weights, biases)
        return x

    def _get_variable(self, name, shape, initializer, weight_decay=0.0, dtype="float32", trainable=True):
        if weight_decay > 0:
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        else:
            regularizer = None
        collections = [tf.GraphKeys.GLOBAL_VARIABLES, RESNET_VARIABLES]
        return tf.get_variable(name, shape, initializer=initializer, dtype=dtype, regularizer=regularizer, collections=collections, trainable=trainable)

    def loss(self, logits, labels):
        #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
        tf.summary.scalar('loss', loss_)
        return loss_

    def add_loss_summaries(self, total_loss):
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        for l in losses + [total_loss]:
            tf.summary.scalar(l.op.name + ' (raw) ', l)
            tf.summary.scalar(l.op.name, loss_averages.average(l))
        return loss_averages_op

    def Train(self, total_loss, global_step, init_lr, optimizer='Adam'):
        #num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        #decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
        decay_steps = LEARNING_RATE_DECAY_STEP
        #lr = tf.train.exponential_decay(init_lr , global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)
        lr = tf.constant(init_lr)
        tf.summary.scalar('learning_rate', lr)
        loss_averages_op = self.add_loss_summaries(total_loss)

        with tf.control_dependencies([loss_averages_op]):
            #opt = tf.train.GradientDescentOptimizer(lr)
            if optimizer == 'RMSProp':
                opt = tf.train.RMSPropOptimizer(lr, decay=0.9, momentum=0.8, epsilon=1e-10, centered=False)
            elif optimizer == 'Adam':
                opt = tf.train.AdamOptimizer(lr)
            else:
                print "No Such Optimizer"
                exit()
            grads = opt.compute_gradients(total_loss)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(
            tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
            train_op = tf.no_op(name='train')
        return train_op



def loss(logits, labels):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]

    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    """Train CIFAR-10 model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
            global_step,
            decay_steps,
            LEARNING_RATE_DECAY_FACTOR,
            staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


