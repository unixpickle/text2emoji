"""
Generate samples from a model.
"""

import argparse

from PIL import Image
import numpy as np
import tensorflow as tf

from text2emoji.data import PLATFORMS
from text2emoji.embed import Embeddings
from text2emoji.model import generate_images

from run_train import checkpoint_name


def main():
    args = arg_parser().parse_args()

    print('Loading embeddings...')
    embeddings = Embeddings(args.embeddings)
    print('Creating model...')
    raw_inputs = tf.placeholder(tf.float32, shape=embeddings.zero_vector().shape)
    inputs = tf.tile(tf.expand_dims(raw_inputs, axis=0), [len(PLATFORMS), 1])
    platforms = tf.one_hot(list(range(len(PLATFORMS))), len(PLATFORMS))
    outputs = generate_images(inputs, platforms)
    outputs = tf.cast((tf.clip_by_value(outputs, -1.0, 1.0) + 1) * 127.5, tf.uint8)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        print('Initializing variables...')
        sess.run(tf.global_variables_initializer())
        print('Restoring from checkpoint...')
        saver.restore(sess, checkpoint_name(args.checkpoint))
        while True:
            phrase = input('Enter phrase: ')
            embedded = embeddings.embed_phrase(phrase)
            print('Producing image...')
            images = sess.run(outputs, feed_dict={raw_inputs: embedded})
            print('Saving image to %s...' % args.output)
            img = Image.fromarray(image_row(images), 'RGBA')
            img.save(args.output)


def image_row(images):
    """
    Convert a batch of images into a horizontal list.
    """
    return np.concatenate([x for x in images], axis=1)


def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--embeddings', help='path to Glove embeddings',
                        default='glove.42B.300d.txt')
    parser.add_argument('--size', help='image size', type=int, default=32)
    parser.add_argument('--checkpoint', help='checkpoint directory', default='checkpoint')
    parser.add_argument('--output', help='output image name', default='output.png')
    return parser


if __name__ == '__main__':
    main()
