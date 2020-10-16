from augmentation.data_types import Augmentation, Data
from ast import literal_eval as make_tuple

import numpy as np
import cv2


class Bias(Augmentation):
    def __init__(self, params):
        self.params = params

    def process(self, data: Data):
        img = data.data['image'].copy()
        img = img.astype('float32')
        img += self.params['b']
        data.data['image'] = img.astype('uint8')


class Gain(Augmentation):
    def __init__(self, params):
        self.params = params

    def process(self, data: Data):
        img = data.data['image'].copy()
        img = img.astype('float32')
        img *= self.params['a']

        data.data['image'] = img.astype('uint8')


class GammaCorrection(Augmentation):
    def __init__(self, params):
        self.params = params

    def process(self, data: Data):
        img = data.data['image'].copy()

        inv_gamma = 1.0 / self.params['gamma']
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)])

        data.data['image'] = cv2.LUT(img.astype(np.uint8), table.astype(np.uint8))


class Noise(Augmentation):

    def process(self, data: Data):
        img = data.data['image']
        noise = np.random.randint(0, 50, (img.shape[0], img.shape[1]))
        jitter = np.zeros_like(img)
        if img.shape[2] == 1:
            jitter[:, :, 1] = noise
        else:
            jitter[:, :, 0] = noise
            jitter[:, :, 1] = noise
            jitter[:, :, 2] = noise

        noise_added = cv2.add(img, jitter)
        data.data['image'] = noise_added


class HistEq(Augmentation):

    def process(self, data: Data):
        img = data.data['image']
        if img.shape[2] == 1:
            img = cv2.equalizeHist(img)
        else:
            img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
            img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
            img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

        data.data['image'] = img


class Tint(Augmentation):
    def __init__(self, params):
        self.params = params
        self.color = make_tuple(self.params['color'])
        self.weight = self.params['weight']

    def process(self, data: Data):
        (h, w, c) = data.data['image'].shape[:3]
        tint_img = np.full((h, w, c), self.color, np.uint8)
        new_img = cv2.addWeighted(data.data['image'], 1 - self.weight, tint_img, self.weight, 0)
        data.data['image'] = new_img
