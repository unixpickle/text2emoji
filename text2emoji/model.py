"""
A generative model for emoji characters.
"""

import tensorflow as tf


def generate_images(embeddings):
    """
    Go from embeddings to images.
    """
    out = tf.layers.dense(embeddings, 256, activation=tf.nn.relu)
    out = tf.reshape(out, [4, 4, 16])

    activation = tf.nn.relu
    out = tf.layers.conv2d(out, 32, 3, activation=activation, padding='same')
    for depth in [128, 64, 32]:
        out = tf.layers.conv2d_transpose(out, depth, 3, activation=activation, padding='same')
    out = tf.layers.conv2d(out, 4, 3, padding='same')
    return out
