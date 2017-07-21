import tensorflow as tf
import numpy as np

IMG_W = 224
IMG_H = 224
CHANNELS = 3

# BGR format
MEAN = [103.939, 116.779, 123.68]

VGG16_WEIGHTS = "data/VGG/vgg16.npy"
VGG19_WEIGHTS = "data/VGG/vgg19.npy"


def generate_VGG16(weights_file=VGG16_WEIGHTS,
                   scope="VGG16_factory",
                   remove_top=False,
                   input_shape=(1, IMG_W, IMG_H, CHANNELS),
                   input_tensor=None,
                   apply_preprocess=True):

    weights = np.load(weights_file, encoding='latin1').item()
    model = {}

    # create a tf.Variable() or use a specific tensor
    # TODO: create a tf.Variable() or a tf.placeholder() ?
    if input_tensor is not None :
        model['input'] = input_tensor
    else:
        with tf.name_scope('input'):
            model['input']= tf.Variable(np.zeros(input_shape), dtype = 'float32')

    # pre-processing step (RGB->BGR + mean subtraction)
    if apply_preprocess:
        with tf.name_scope("preprocessing"):
            red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=model['input'])
            bgr = tf.concat(axis=3, values=[blue - MEAN[0], green - MEAN[1], red - MEAN[2], ])
            model['preprocess'] = bgr
    else:
        model['preprocess'] = model['input']

    if isinstance(scope, str):
        # the scope is a string, we can create a  new scope, named by 'scope' (str).
        # this scope will be returned for reuse (variable sharing)
        with tf.variable_scope(scope) as new_scope:

            model['conv1_1'] = _conv2d_relu(model['preprocess'], 'conv1_1', weights)
            model['conv1_2'] = _conv2d_relu(model['conv1_1'], 'conv1_2', weights)
            model['pool_1'] = _pool(model['conv1_2'], 'pool_1')

            model['conv2_1'] = _conv2d_relu(model['pool_1'], 'conv2_1', weights)
            model['conv2_2'] = _conv2d_relu(model['conv2_1'], 'conv2_2', weights)
            model['pool_2'] = _pool(model['conv2_2'], 'pool_2')

            model['conv3_1'] = _conv2d_relu(model['pool_2'], 'conv3_1', weights)
            model['conv3_2'] = _conv2d_relu(model['conv3_1'], 'conv3_2', weights)
            model['conv3_3'] = _conv2d_relu(model['conv3_2'], 'conv3_3', weights)
            model['pool_3'] = _pool(model['conv3_3'], 'pool_3')

            model['conv4_1'] = _conv2d_relu(model['pool_3'], 'conv4_1', weights)
            model['conv4_2'] = _conv2d_relu(model['conv4_1'], 'conv4_2', weights)
            model['conv4_3'] = _conv2d_relu(model['conv4_2'], 'conv4_3', weights)
            model['pool_4'] = _pool(model['conv4_3'], 'pool_4')

            model['conv5_1'] = _conv2d_relu(model['pool_4'], 'conv5_1', weights)
            model['conv5_2'] = _conv2d_relu(model['conv5_1'], 'conv5_2', weights)
            model['conv5_3'] = _conv2d_relu(model['conv5_2'], 'conv5_3', weights)
            model['pool_5'] = _pool(model['conv5_3'], 'pool_5')

            if remove_top:
                return model, new_scope

            model['flatten'] = _flatten(model['pool_5'])
            model['fc6'] = _fc_relu(model['flatten'], 'fc6', weights)
            model['fc7'] = _fc_relu(model['fc6'], 'fc7', weights)
            model['fc8'] = _fc_linear(model['fc7'], 'fc8', weights)

            model['prob'] = _prob(model['fc8'], 'prob')

            return model, new_scope

    elif isinstance(scope, tf.VariableScope):
        # the scope is a tf.VariableScope
        with tf.variable_scope(scope, reuse=True):

            model['conv1_1'] = _conv2d_relu(model['preprocess'], 'conv1_1', weights, reuse_scope=True)
            model['conv1_2'] = _conv2d_relu(model['conv1_1'], 'conv1_2', weights, reuse_scope=True)
            model['pool_1'] = _pool(model['conv1_2'], 'pool_1')

            model['conv2_1'] = _conv2d_relu(model['pool_1'], 'conv2_1', weights, reuse_scope=True)
            model['conv2_2'] = _conv2d_relu(model['conv2_1'], 'conv2_2', weights, reuse_scope=True)
            model['pool_2'] = _pool(model['conv2_2'], 'pool_2')

            model['conv3_1'] = _conv2d_relu(model['pool_2'], 'conv3_1', weights, reuse_scope=True)
            model['conv3_2'] = _conv2d_relu(model['conv3_1'], 'conv3_2', weights, reuse_scope=True)
            model['conv3_3'] = _conv2d_relu(model['conv3_2'], 'conv3_3', weights, reuse_scope=True)
            model['pool_3'] = _pool(model['conv3_3'], 'pool_3')

            model['conv4_1'] = _conv2d_relu(model['pool_3'], 'conv4_1', weights, reuse_scope=True)
            model['conv4_2'] = _conv2d_relu(model['conv4_1'], 'conv4_2', weights, reuse_scope=True)
            model['conv4_3'] = _conv2d_relu(model['conv4_2'], 'conv4_3', weights, reuse_scope=True)
            model['pool_4'] = _pool(model['conv4_3'], 'pool_4')

            model['conv5_1'] = _conv2d_relu(model['pool_4'], 'conv5_1', weights, reuse_scope=True)
            model['conv5_2'] = _conv2d_relu(model['conv5_1'], 'conv5_2', weights, reuse_scope=True)
            model['conv5_3'] = _conv2d_relu(model['conv5_2'], 'conv5_3', weights, reuse_scope=True)
            model['pool_5'] = _pool(model['conv5_3'], 'pool_5')

            if remove_top:
                return model, scope
            # TODO : issue when creating a first scope with 'remove_top=True' and the second has 'remove_top=False'
            model['flatten'] = _flatten(model['pool_5'])
            model['fc6'] = _fc_relu(model['flatten'], 'fc6', weights, reuse_scope=True)
            model['fc7'] = _fc_relu(model['fc6'], 'fc7', weights, reuse_scope=True)
            model['fc8'] = _fc_linear(model['fc7'], 'fc8', weights, reuse_scope=True)

            model['prob'] = _prob(model['fc8'], 'prob')

            return model, scope

    else:
        print("Invalid scope")
        exit(-1)


def generate_VGG19(weights_file=VGG19_WEIGHTS,
                   scope="VGG19_factory",
                   remove_top=False,
                   input_shape=(1, IMG_W, IMG_H, CHANNELS),
                   input_tensor=None,
                   apply_preprocess=True):

    weights = np.load(weights_file, encoding='latin1').item()
    model = {}

    # create a tf.Variable() or use a specific tensor
    if input_tensor is not None:
        model['input'] = input_tensor
    else:
        with tf.name_scope('input'):
            model['input'] = tf.Variable(np.zeros(input_shape), dtype='float32')

    # pre-processing step (RGB->BGR + mean subtraction)
    if apply_preprocess:
        with tf.name_scope("preprocessing"):
            red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=model['input'])
            bgr = tf.concat(axis=3, values=[blue - MEAN[0], green - MEAN[1], red - MEAN[2], ])
            model['preprocess'] = bgr
    else:
        model['preprocess'] = model['input']

    if isinstance(scope, str):

        with tf.variable_scope(scope) as new_scope:
            model['conv1_1'] = _conv2d_relu(model['preprocess'], 'conv1_1', weights)
            model['conv1_2'] = _conv2d_relu(model['conv1_1'], 'conv1_2', weights)
            model['pool_1'] = _pool(model['conv1_2'], 'pool_1')

            model['conv2_1'] = _conv2d_relu(model['pool_1'], 'conv2_1', weights)
            model['conv2_2'] = _conv2d_relu(model['conv2_1'], 'conv2_2', weights)
            model['pool_2'] = _pool(model['conv2_2'], 'pool_2')

            model['conv3_1'] = _conv2d_relu(model['pool_2'], 'conv3_1', weights)
            model['conv3_2'] = _conv2d_relu(model['conv3_1'], 'conv3_2', weights)
            model['conv3_3'] = _conv2d_relu(model['conv3_2'], 'conv3_3', weights)
            model['conv3_4'] = _conv2d_relu(model['conv3_3'], 'conv3_4', weights)
            model['pool_3'] = _pool(model['conv3_4'], 'pool_3')

            model['conv4_1'] = _conv2d_relu(model['pool_3'], 'conv4_1', weights)
            model['conv4_2'] = _conv2d_relu(model['conv4_1'], 'conv4_2', weights)
            model['conv4_3'] = _conv2d_relu(model['conv4_2'], 'conv4_3', weights)
            model['conv4_4'] = _conv2d_relu(model['conv4_3'], 'conv4_4', weights)
            model['pool_4'] = _pool(model['conv4_4'], 'pool_4')

            model['conv5_1'] = _conv2d_relu(model['pool_4'], 'conv5_1', weights)
            model['conv5_2'] = _conv2d_relu(model['conv5_1'], 'conv5_2', weights)
            model['conv5_3'] = _conv2d_relu(model['conv5_2'], 'conv5_3', weights)
            model['conv5_4'] = _conv2d_relu(model['conv5_3'], 'conv5_4', weights)
            model['pool_5'] = _pool(model['conv5_3'], 'pool_5')

            if remove_top:
                return model, new_scope

            model['flatten'] = _flatten(model['pool_5'])
            model['fc6'] = _fc_relu(model['flatten'], 'fc6', weights)
            model['fc7'] = _fc_relu(model['fc6'], 'fc7', weights)
            model['fc8'] = _fc_linear(model['fc7'], 'fc8', weights)
            model['prob'] = _prob(model['fc8'], 'prob')

            return model, new_scope

    elif isinstance(scope, tf.VariableScope):
        with tf.variable_scope(scope, reuse=True):
            model['conv1_1'] = _conv2d_relu(model['preprocess'], 'conv1_1', weights)
            model['conv1_2'] = _conv2d_relu(model['conv1_1'], 'conv1_2', weights)
            model['pool_1'] = _pool(model['conv1_2'], 'pool_1')

            model['conv2_1'] = _conv2d_relu(model['pool_1'], 'conv2_1', weights)
            model['conv2_2'] = _conv2d_relu(model['conv2_1'], 'conv2_2', weights)
            model['pool_2'] = _pool(model['conv2_2'], 'pool_2')

            model['conv3_1'] = _conv2d_relu(model['pool_2'], 'conv3_1', weights)
            model['conv3_2'] = _conv2d_relu(model['conv3_1'], 'conv3_2', weights)
            model['conv3_3'] = _conv2d_relu(model['conv3_2'], 'conv3_3', weights)
            model['conv3_4'] = _conv2d_relu(model['conv3_3'], 'conv3_4', weights)
            model['pool_3'] = _pool(model['conv3_4'], 'pool_3')

            model['conv4_1'] = _conv2d_relu(model['pool_3'], 'conv4_1', weights)
            model['conv4_2'] = _conv2d_relu(model['conv4_1'], 'conv4_2', weights)
            model['conv4_3'] = _conv2d_relu(model['conv4_2'], 'conv4_3', weights)
            model['conv4_4'] = _conv2d_relu(model['conv4_3'], 'conv4_4', weights)
            model['pool_4'] = _pool(model['conv4_4'], 'pool_4')

            model['conv5_1'] = _conv2d_relu(model['pool_4'], 'conv5_1', weights)
            model['conv5_2'] = _conv2d_relu(model['conv5_1'], 'conv5_2', weights)
            model['conv5_3'] = _conv2d_relu(model['conv5_2'], 'conv5_3', weights)
            model['conv5_4'] = _conv2d_relu(model['conv5_3'], 'conv5_4', weights)
            model['pool_5'] = _pool(model['conv5_3'], 'pool_5')

            if remove_top:
                return model, scope

            model['flatten'] = _flatten(model['pool_5'])
            model['fc6'] = _fc_relu(model['flatten'], 'fc6', weights)
            model['fc7'] = _fc_relu(model['fc6'], 'fc7', weights)
            model['fc8'] = _fc_linear(model['fc7'], 'fc8', weights)
            model['prob'] = _prob(model['fc8'], 'prob')

            return model, scope

    else:
        print("Invalid scope")
        exit(-1)


def _get_weights(layer_name, weights):
    """
    Load weights with name 'layer_name'
    weights[layer_name][0] : W (kernel conv or matrix)
    weights[layer_name][1] : b (bias vector)
    """
    W = weights[layer_name][0]
    b = weights[layer_name][1]
    return W, b


def _conv2d_relu(prev_layer, layer_name, weights, reuse_scope=False):
    """
    Return the Conv2D + RELU layer using the weights, biases from the VGG
    model at 'layer'.
    """
    with tf.name_scope(layer_name):
        if reuse_scope is False:
            w_np, b_np = _get_weights(layer_name, weights)

            with tf.variable_scope(layer_name):
                w = tf.get_variable('W', shape=tuple(w_np.shape),
                                    dtype=w_np.dtype, trainable=False,
                                    initializer=tf.constant_initializer(w_np))

                b = tf.get_variable('b', shape=tuple(b_np.shape),
                                    dtype=b_np.dtype, trainable=False,
                                    initializer=tf.constant_initializer(b_np))

        else:
            with tf.variable_scope(layer_name, reuse=True):
                w = tf.get_variable('W')
                b = tf.get_variable('b')

        conv = tf.nn.conv2d(prev_layer, filter=w, strides=[1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, b)
        acti = tf.nn.relu(out, name=layer_name)
        return acti


def _pool(prev_layer, layer_name):
    """
    Return the MaxPooling layer.
    """
    with tf.name_scope(layer_name):
        return tf.nn.max_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def _flatten(prev_layer):
    """
    Reshape layer, flatten operation.
    """

    with tf.name_scope('flatten'):
        shape = int(np.prod(prev_layer.get_shape()[1:]))
        return tf.reshape(prev_layer, [-1, shape])


def _fc_relu(prev_layer, layer_name, weights, reuse_scope=False):
    """
    Return the Dense/Fully Connected  + ReLU layer using the weights, biases from the VGG model
    """

    with tf.name_scope(layer_name):
        if reuse_scope is False:
            w_np, b_np = _get_weights(layer_name, weights)

            with tf.variable_scope(layer_name):
                w = tf.get_variable('W', shape=tuple(w_np.shape),
                                dtype=w_np.dtype, trainable=False,
                                initializer=tf.constant_initializer(w_np))

                b = tf.get_variable('b', shape=tuple(b_np.shape),
                                dtype=b_np.dtype, trainable=False,
                                initializer=tf.constant_initializer(b_np))

        else:
            with tf.variable_scope(layer_name, reuse=True):
                w = tf.get_variable('W')
                b = tf.get_variable('b')
        return tf.nn.relu(tf.nn.bias_add(tf.matmul(prev_layer, w), b))


def _fc_linear(prev_layer, layer_name, weights, reuse_scope=False):
    """
    Return the Dense/Fully Connected layer using the weights, biases from the VGG model
    """
    with tf.name_scope(layer_name):
        if reuse_scope is False:
            w_np, b_np = _get_weights(layer_name, weights)

            with tf.variable_scope(layer_name):
                w = tf.get_variable('W', shape=tuple(w_np.shape),
                                dtype=w_np.dtype, trainable=False,
                                initializer=tf.constant_initializer(w_np))

                b = tf.get_variable('b', shape=tuple(b_np.shape),
                                dtype=b_np.dtype, trainable=False,
                                initializer=tf.constant_initializer(b_np))

        else:
            with tf.variable_scope(layer_name, reuse=True):
                w = tf.get_variable('W')
                b = tf.get_variable('b')
        return tf.nn.bias_add(tf.matmul(prev_layer, w), b)


def _prob(prev_layer, layer_name):
    """
    Returns the softmax.
    """
    with tf.name_scope(layer_name):
        return tf.nn.softmax(prev_layer)
