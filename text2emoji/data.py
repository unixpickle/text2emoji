"""
Loading the emoji dataset.
"""

import json
import os

import tensorflow as tf


def create_dataset(embeddings, data_dir, image_size):
    """
    Create a TensorFlow dataset with input embeddings and
    output images corresponding to the embeddings.
    """
    emojis = _emoji_paths(data_dir)
    all_embeddings = []
    all_paths = []
    for name, paths in emojis:
        embedding = embeddings.embed_phrase(name)
        all_embeddings.extend([embedding] * len(paths))
        all_paths.extend(paths)
    dataset = tf.data.Dataset.from_tensor_slices((all_embeddings, all_paths))

    def load_image(embedding, path):
        data = tf.read_file(path)
        raw_image = tf.image.decode_image(data)
        raw_image.set_shape([None] * 3)
        resized = tf.image.resize_images(raw_image, [image_size] * 2)
        return embedding, resized

    return dataset.flat_map(load_image)


def _emoji_paths(data_dir):
    """
    Produces a dict mapping emoji names to lists of image
    paths.
    """
    res = {}
    with open(os.path.join(data_dir, 'text.json'), 'r') as in_file:
        data = json.loads(in_file)
        for codepoints, name in data.items():
            codepoint_dir = os.path.join(data_dir, codepoints)
            res[name] = [os.path.join(codepoint_dir, x) for x in os.listdir(codepoint_dir)
                         if x.endswith('.png')]
    return res
