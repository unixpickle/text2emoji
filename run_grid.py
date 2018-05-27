"""
Create a grid of reconstructions for some emojis.
"""

import argparse

from PIL import Image
import numpy as np
import tensorflow as tf

from text2emoji.data import PLATFORMS
from text2emoji.embed import Embeddings
from text2emoji.model import generate_images

from run_train import checkpoint_name


PHRASES = ['thinking face', 'thumbs up', 'smiling face with heart-eyes', 'pile of poo',
           'robot face', 'woman guard', 'brain', 'red heart', 'giraffe', 'dolphin',
           'hamburger', 'airplane departure', 'rainbow', 'cross mark']


def main():
    args = arg_parser().parse_args()

    print('Loading embeddings...')
    embeddings = Embeddings(args.embeddings)
    print('Creating inputs')
    # A batch of inputs with words on the major axis and
    # platform on the minor axis.
    embeddings = [x
                  for phrase in PHRASES
                  for x in [embeddings.embed_phrase(phrase)] * len(PLATFORMS)]
    platforms = tf.one_hot(list(range(len(PLATFORMS))) * len(PHRASES), len(PLATFORMS))
    print('Creating model...')
    outputs = generate_images(tf.constant(np.array(embeddings)), platforms)
    outputs = tf.cast((tf.clip_by_value(outputs, -1.0, 1.0) + 1) * 127.5, tf.uint8)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        print('Initializing variables...')
        sess.run(tf.global_variables_initializer())
        print('Restoring from checkpoint...')
        saver.restore(sess, checkpoint_name(args.checkpoint))
        print('Running model...')
        outputs = sess.run(outputs)
        grid = np.zeros([args.size * len(PHRASES), args.size * len(PLATFORMS), 4], dtype='uint8')
        for row in range(len(PHRASES)):
            for col in range(len(PLATFORMS)):
                grid[row*args.size: (row + 1) * args.size,
                     col*args.size: (col + 1) * args.size] = outputs[row * len(PLATFORMS) + col]
        img = Image.fromarray(grid, 'RGBA')
        img.save(args.output)


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
