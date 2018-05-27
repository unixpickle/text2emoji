"""
A generative model for emoji characters.
"""

import tensorflow as tf

from .data import PLATFORMS


def generate_images(embeddings, platforms, training=False):
    """
    Go from embeddings to images.
    """
    plat_embed = tf.get_variable('platform_embed',
                                 initializer=tf.truncated_normal_initializer(),
                                 shape=[len(PLATFORMS), embeddings.get_shape()[-1].value])
    embeddings = embeddings + tf.matmul(platforms, plat_embed)

    out = tf.layers.dense(embeddings, 256, activation=tf.nn.relu)
    out = tf.reshape(out, [-1, 4, 4, 16])

    def activation(x):
        out = tf.layers.batch_normalization(x, training=training)
        out = tf.nn.relu(out)
        return out

    out = tf.layers.conv2d(out, 64, 3, activation=activation, padding='same')
    for depth in [128, 64, 32]:
        out_1 = tf.layers.conv2d(out, depth, 3, activation=activation, padding='same')
        out_2 = tf.layers.conv2d(out_1, depth, 3, padding='same')
        out_2 = tf.layers.batch_normalization(out_2, training=training)
        out = tf.nn.relu(out + out_2)
        out = tf.layers.conv2d_transpose(out, depth, 3, strides=2, activation=activation,
                                         padding='same')
    out = tf.layers.conv2d(out, 4, 3, padding='same')
    return out
