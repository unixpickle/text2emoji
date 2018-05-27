"""
Loading the emoji dataset.
"""

import json
import os

import numpy as np
import tensorflow as tf

PLATFORMS = [x + '.png' for x in ['apple', 'fb', 'google', 'one', 'samsung', 'twitter', 'windows']]


def create_dataset(embeddings, data_dir, image_size):
    """
    Create a Dataset of (embedding, platform, input):
      embedding: a phrase embedding for the emoji.
      platform: a one-hot platform vector
      input: the input RGBA float32 image.
    """
    emojis = _emoji_paths(data_dir)
    all_embeddings = []
    all_platforms = []
    all_paths = []
    for name, paths in emojis.items():
        embedding = embeddings.embed_phrase(name)
        all_embeddings.extend([embedding] * len(paths))
        all_paths.extend(paths)
        for path in paths:
            idx = PLATFORMS.index(os.path.basename(path))
            vec = np.zeros([len(PLATFORMS)], dtype='float32')
            vec[idx] = 1
            all_platforms.append(vec)
    dataset = tf.data.Dataset.from_tensor_slices((np.array(all_embeddings),
                                                  np.array(all_platforms),
                                                  all_paths))

    def load_image(embedding, platform, path):
        data = tf.read_file(path)
        raw_image = tf.image.decode_png(data, channels=4)
        raw_image.set_shape([None, None, 4])
        resized = tf.image.resize_images(raw_image, [image_size] * 2)
        floats = tf.cast(resized, tf.float32) / 127.5 - 1
        return embedding, platform, floats

    return dataset.map(load_image)


def _emoji_paths(data_dir):
    """
    Produces a dict mapping emoji names to lists of image
    paths.
    """
    res = {}
    with open(os.path.join(data_dir, 'text.json'), 'r') as in_file:
        data = json.load(in_file)
    for codepoints, name in data.items():
        codepoint_dir = os.path.join(data_dir, codepoints)
        res[name] = [os.path.join(codepoint_dir, x) for x in os.listdir(codepoint_dir)
                     if x.endswith('.png')]
    return res
