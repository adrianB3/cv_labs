from augmentation.data_types import Augmentation, Data
from ast import literal_eval as make_tuple

import numpy as np
import cv2


class Luminosity(Augmentation):
    def __init__(self, params):
        self.params = params

    def process(self, data: Data):
        img = data.data['image'].copy()
        img = img.astype('float32')
        img += self.params['bias']
        data.data['image'] = img.astype('uint8')


class Contrast(Augmentation):
    def __init__(self, params):
        self.params = params

    def process(self, data: Data):
        img = data.data['image'].copy()
        img = img.astype('float32')
        if self.params['gain'] != 0:
            img *= self.params['gain']

        data.data['image'] = img.astype('uint8')


class GammaCorrection(Augmentation):
    def __init__(self, params):
        self.params = params

    def process(self, data: Data):
        img = data.data['image'].copy()

        inv_gamma = 1.0 / self.params['gamma']
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)])
        corrected_image = cv2.LUT(img.astype(np.uint8), table.astype(np.uint8))

        data.data['image'] = corrected_image


class Noise(Augmentation):
    def __init__(self, params):
        self.params = params

    def process(self, data: Data):
        img = data.data['image']
        if img.ndim == 3:
            h, w, c = img.shape
        elif img.ndim == 2:
            h, w = img.shape
        if self.params['type'] == "gaussian":
            mean = self.params['gaussian']['mean']
            var = self.params['gaussian']['var']
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (h, w, c))
            gauss = gauss.reshape(h, w, c)
            noisy = img + gauss
            data.data['image'] = noisy

        if self.params['type'] == "sp":
            svsp = 0.5
            amount = self.params['sp']['amount']
            # Salt mode
            num_salt = np.ceil(amount * img.size * svsp)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in img.shape]
            img[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * img.size * (1. - svsp))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in img.shape]
            img[coords] = 0
            data.data['image'] = img

        if self.params['type'] == "poisson":
            vals = len(np.unique(img))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(img * vals) / float(vals)
            data.data['image'] = noisy

        if self.params['type'] == "speckle":
            gauss = np.random.randn(h, w, c)
            gauss = gauss.reshape(h, w, c)
            noisy = img + img * gauss
            data.data['image'] = noisy


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


class RandomErasing(Augmentation):
    def __init__(self, params):
        self.params = params

    def process(self, data: Data):
        input_img = data.data['image']
        if input_img.ndim == 3:
            h, w, c = input_img.shape
        elif input_img.ndim == 2:
            h, w = input_img.shape

        prob = np.random.rand()  # returns a random nb between 0 and 1
        if self.params['probability'] < prob:
            pass
        else:
            while True:
                s = np.random.uniform(self.params['min_prop'], self.params['max_prop']) * h * w
                r = np.random.uniform(self.params['min_ratio'], self.params['max_ratio'])
                w1 = int(np.sqrt(s / r))
                h1 = int(np.sqrt(s * r))
                left = np.random.randint(0, w)
                top = np.random.randint(0, h)

                if left + w1 <= w and top + h1 <= h:
                    break

            if input_img.ndim == 3:
                col = np.random.uniform(0, 255, (h1, w1, c))
            if input_img.ndim == 2:
                col = np.random.uniform(0, 255, (h1, w1))

            input_img[top:top + h1, left:left + w1] = col
            data.data['image'] = input_img
