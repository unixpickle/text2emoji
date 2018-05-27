"""
Train a model.
"""

import argparse
import os

import tensorflow as tf

from text2emoji.data import create_dataset
from text2emoji.embed import Embeddings
from text2emoji.model import generate_images


def main():
    args = arg_parser().parse_args()

    print('Loading embeddings...')
    embeddings = Embeddings(args.embeddings)
    print('Creating dataset...')
    try:
        dataset = create_dataset(embeddings, args.data_dir, args.size)
    finally:
        embeddings.close()
    dataset = dataset.shuffle(10000).repeat().batch(args.batch_size)
    embeddings, images = dataset.make_one_shot_iterator().get_next()

    print('Creating model...')
    reconstructions = generate_images(embeddings)
    loss = tf.reduce_mean(tf.abs(reconstructions - images))
    minimize = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(loss)

    cur_step = tf.Variable(initial_value=tf.constant(0), name='global_step', trainable=False)
    inc_step = tf.assign_add(cur_step, tf.constant(1))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        print('Initializing variables...')
        sess.run(tf.global_variables_initializer())
        if os.path.exists(args.checkpoint):
            print('Restoring from checkpoint...')
            saver.restore(sess, checkpoint_name(args.checkpoint))
        print('Training...')
        while True:
            cur_loss, cur_step, _ = sess.run([loss, inc_step, minimize])
            print('step %d: loss=%f' % (cur_step, cur_loss))
            if cur_step % args.save_interval == 0:
                if not os.path.exists(args.checkpoint):
                    os.mkdir(args.checkpoint)
                saver.save(sess, checkpoint_name(args.checkpoint))


def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--embeddings', help='path to Glove embeddings',
                        default='glove.42B.300d.txt')
    parser.add_argument('--data-dir', help='path to emoji data', default='emoji_data')
    parser.add_argument('--size', help='image size', type=int, default=32)
    parser.add_argument('--lr', help='training step size', type=float, default=1e-3)
    parser.add_argument('--batch-size', help='SGD batch size', type=int, default=32)
    parser.add_argument('--checkpoint', help='checkpoint directory', default='checkpoint')
    parser.add_argument('--save-interval', help='iterations per save', type=int, default=100)
    return parser


def checkpoint_name(dir_name):
    return os.path.join(dir_name, 'model.ckpt')


if __name__ == '__main__':
    main()
