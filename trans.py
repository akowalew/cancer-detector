import numpy as np

class LogSig:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, y):
        return y * (1 - y)

class TanSig:
    def __call__(self, x):
        return np.tanh(x)

    def gradient(self, y):
        return 1.0 - np.square(y)