# cnn_autoencoder.py
"""CNN autoencoder model definition."""
from typing import Sequence

import tensorflow.compat.v2 as tf
from tensorflow.compat.v2 import keras

import model_utils


def encoder(
        input_tensor,
        layers_list,
        pool_list,
        dropout=model_utils.DROPOUT_DEFAULT,
        batch_norm=model_utils.BATCH_NORM_DEFAULT,
        l1_regularization=model_utils.L1_REGULARIZATION_DEFAULT,
        l2_regularization=model_utils.L2_REGULARIZATION_DEFAULT
):
    net = input_tensor
    net = model_utils.conv2d_layer(
        filters=layers_list[0],
        l1_regularization=l1_regularization,
        l2_regularization=l2_regularization)(net)
    if batch_norm == 'all':
        net = keras.layers.BatchNormalization()(net)
    net = keras.layers.LeakyReLU()(net)
    net = keras.layers.Dropout(dropout)(net)
    net = model_utils.conv2d_layer(
        filters=layers_list[0],
        l1_regularization=l1_regularization,
        l2_regularization=l2_regularization)(net)
    shortcut = model_utils.conv2d_layer(
        filters=layers_list[0],
        kernel_size=model_utils.RES_SHORTCUT_KERNEL_SIZE,
        l1_regularization=l1_regularization,
        l2_regularization=l2_regularization)(input_tensor)
    if batch_norm == 'all':
        shortcut = keras.layers.BatchNormalization()(shortcut)
    shortcut = keras.layers.Dropout(dropout)(shortcut)

    net = shortcut + net

    for i in range(len(layers_list[1:])):
        filters = layers_list[1 + i]
        pool = pool_list[1 + i]
        net = model_utils.res_block(
            net,
            filters=(filters, filters),
            strides=(pool if pool > 1 else 1, 1),  # Only downsample if pool > 1
            pool_size=pool,
            dropout=dropout,
            batch_norm=batch_norm,
            l1_regularization=l1_regularization,
            l2_regularization=l2_regularization)
    return net


def decoder_block(
        input_tensor,
        filters,
        pool_size,
        dropout=model_utils.DROPOUT_DEFAULT,
        batch_norm=model_utils.BATCH_NORM_DEFAULT,
        l1_regularization=model_utils.L1_REGULARIZATION_DEFAULT,
        l2_regularization=model_utils.L2_REGULARIZATION_DEFAULT
):
    net = keras.layers.UpSampling2D(
        size=(pool_size, pool_size), interpolation='nearest')(input_tensor)
    net = model_utils.res_block(
        net,
        filters=(filters, filters),
        strides=model_utils.RES_DECODER_STRIDES,  # (1, 1)
        pool_size=pool_size,
        dropout=dropout,
        batch_norm=batch_norm,
        l1_regularization=l1_regularization,
        l2_regularization=l2_regularization)
    return net


def decoder(
        input_tensor,
        layers_list,
        pool_list,
        dropout=model_utils.DROPOUT_DEFAULT,
        batch_norm=model_utils.BATCH_NORM_DEFAULT,
        l1_regularization=model_utils.L1_REGULARIZATION_DEFAULT,
        l2_regularization=model_utils.L2_REGULARIZATION_DEFAULT
):
    net = input_tensor
    for layer, pool in zip(layers_list, pool_list):
        net = decoder_block(net, layer, pool, dropout, batch_norm,
                            l1_regularization, l2_regularization)
    return net


def create_model(
        input_tensor,
        num_out_channels,
        encoder_layers,
        decoder_layers,
        encoder_pools,
        decoder_pools,
        dropout=model_utils.DROPOUT_DEFAULT,
        batch_norm=model_utils.BATCH_NORM_DEFAULT,
        l1_regularization=model_utils.L1_REGULARIZATION_DEFAULT,
        l2_regularization=model_utils.L2_REGULARIZATION_DEFAULT
):
    if len(encoder_layers) != len(encoder_pools):
        raise ValueError('Length of encoder_layers and encoder_pools should be equal.')
    if len(decoder_layers) != len(decoder_pools):
        raise ValueError('Length of decoder_layers and decoder_pools should be equal.')
    if len(decoder_layers) > len(encoder_layers):
        raise ValueError('Length of decoder_layers should be <= length of encoder_layers.')

    # Encoder
    bottleneck_x = encoder(input_tensor, encoder_layers, encoder_pools, dropout,
                           batch_norm, l1_regularization, l2_regularization)

    # Bottleneck (no downsampling)
    x = model_utils.res_block(
        bottleneck_x,
        filters=(encoder_layers[-1], encoder_layers[-1]),
        strides=(1, 1),  # Explicitly no pooling
        pool_size=1,  # No pooling
        dropout=dropout,
        batch_norm=batch_norm,
        l1_regularization=l1_regularization,
        l2_regularization=l2_regularization)


    # Decoder
    x = decoder(x, decoder_layers, decoder_pools, dropout, batch_norm,
                l1_regularization, l2_regularization)

    # Final layer
    x = model_utils.conv2d_layer(
        filters=num_out_channels,
        kernel_size=model_utils.RES_SHORTCUT_KERNEL_SIZE,
        l1_regularization=l1_regularization,
        l2_regularization=l2_regularization)(x)

    model = keras.Model(inputs=input_tensor, outputs=x)
    return model
