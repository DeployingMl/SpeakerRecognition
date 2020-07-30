from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Conv1D, MaxPool1D, \
    Lambda, Activation, Dropout, Input, Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


def preprocess(x):
    x = (x + 0.8) / 7.0
    x = K.clip(x, -5, 5)
    return x


def preprocess_raw(x):
    return x


Preprocess = Lambda(preprocess)

PreprocessRaw = Lambda(preprocess_raw)


def relu6(x):
    return K.relu(x, max_value=6)


def conv_1d_time_stacked_model(input_size=32000, num_classes=5):
    """ Creates a 1D model for temporal data.

    Note: Use only
    with compute_mfcc = False (e.g. raw waveform data).
    Args:
        input_size: How big the input vector is.
        num_classes: How many classes are to be recognized.

    Returns:
        Compiled keras model
    """
    input_layer = Input(shape=(1, input_size))
    x = input_layer
    x = Reshape([800, 40])(x)
    x = PreprocessRaw(x)

    def _reduce_conv(x, num_filters, k, strides=2, padding='valid'):
        x = Conv1D(
            num_filters,
            k,
            padding=padding,
            use_bias=False,
            kernel_regularizer=l2(0.00001))(
                x)
        x = BatchNormalization()(x)
        x = Activation(relu6)(x)
        x = MaxPool1D(pool_size=3, strides=strides, padding=padding)(x)
        return x

    def _context_conv(x, num_filters, k, dilation_rate=1, padding='valid'):
        x = Conv1D(
            num_filters,
            k,
            padding=padding,
            dilation_rate=dilation_rate,
            kernel_regularizer=l2(0.00001),
            use_bias=False)(
                x)
        x = BatchNormalization()(x)
        x = Activation(relu6)(x)
        return x

    x = _context_conv(x, 32, 1)
    x = _reduce_conv(x, 48, 3)
    x = _context_conv(x, 48, 3)
    x = _reduce_conv(x, 96, 3)
    x = _context_conv(x, 96, 3)
    x = _reduce_conv(x, 128, 3)
    x = _context_conv(x, 128, 3)
    x = _reduce_conv(x, 160, 3)
    x = _context_conv(x, 160, 3)
    x = _reduce_conv(x, 192, 3)
    x = _context_conv(x, 192, 3)
    x = _reduce_conv(x, 256, 3)
    x = _context_conv(x, 256, 3)

    x = Dropout(0.3)(x)
    x = Conv1D(num_classes, 5, activation='softmax')(x)
    x = Reshape([-1])(x)

    model = Model(input_layer, x, name='conv_1d_time_stacked')
    model.compile(
        optimizer=keras.optimizers.Adam(lr=3e-4),
        loss=keras.losses.categorical_crossentropy,
        metrics=[keras.metrics.categorical_accuracy])
    return model


def speech_model(input_size=32000, num_classes=5):
    model = conv_1d_time_stacked_model(input_size=input_size,
                                       num_classes=num_classes)
    return model
