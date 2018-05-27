"""
Word embeddings.
"""

import numpy as np


class Embeddings:
    """
    A handle on a Glove word embedding table.

    Word embedding tables are lines where the first token
    is a word, and the remaining tokens are floats.
    """

    def __init__(self, path):
        self._offsets = {}
        offset = 0
        with open(path, 'rb') as in_file:
            for line in in_file:
                self._offsets[line.decode('utf-8').split(' ')[0]] = offset
                offset += len(line)
        self._file = open(path, 'r')

    def lookup(self, word):
        """
        Lookup the word vector for a word.
        """
        if word not in self._offsets:
            # No unknown token, as far as I can tell :\
            return self.zero_vector()
        self._file.seek(self._offsets[word])
        line = self._file.readline().strip().split(' ')
        return np.array([float(x) for x in line[1:]], dtype='float32')

    def embed_phrase(self, phrase):
        """
        Embed a phrase into a mean word vector.
        """
        letters = 'abcdefghijklmnopqrstuvwxyz'
        phrase = ''.join((x if x in letters else ' ') for x in phrase.lower())
        total = self.zero_vector()
        for token in phrase.lower().split():
            total += self.lookup(token)
        return total

    def zero_vector(self):
        """
        Generate an all-zero embedding.
        """
        return np.zeros_like(self.lookup('a'))

    def close(self):
        """
        Close the underlying file.
        """
        self._file.close()
