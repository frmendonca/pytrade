import numpy as np

class DualTransform():
    def __init__(self, L, H):
        self._L = L
        self._H = H

    def transform(self, data):
        return self._L - self._H*np.log((self._H-data)/(self._H - self._L))

    def inverse_transform(self, data):
        return self._H - np.exp((self._L - data)/self._H)*(self._H - self._L)