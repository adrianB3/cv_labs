from augmentation.data_types import Augmentation, Data
import numpy as np

class Bias(Augmentation):
    def __init__(self, params):
        self.params = params

    def process(self, data: Data):
        data.data['image'][:, :, :] += self.params['b']


class Gain(Augmentation):
    def __init__(self, params):
        self.params = params

    def process(self, data: Data):
        data.data['image'][:, :, :] *= self.params['a']


class GammaCorrection(Augmentation):
    def __init__(self, params):
        self.params = params

    def process(self, data: Data):
        data.data['image'][:, :, :] = np.power(data.data['image'][:, :, :], int(1/self.params['gamma']))
