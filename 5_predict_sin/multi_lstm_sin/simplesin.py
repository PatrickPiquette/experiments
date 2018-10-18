import numpy as np


class SimpleSin:
    def __init__(self):
        self.offset = 0

    def sin(self, X, signal_freq=60.):
        return np.sin(2 * np.pi * X / signal_freq)

    def sample(self, sample_size):
        X = np.arange(sample_size)
        X += self.offset
        Y = self.sin(X)
        self.offset += sample_size
        return X, Y
