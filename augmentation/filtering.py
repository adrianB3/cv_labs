import cv2
import numpy as np

from augmentation.data_types import Augmentation, Data
from ast import literal_eval as make_tuple


class Blur(Augmentation):
    def __init__(self, params):
        self.params = params

    def process(self, data: Data):
        kernel_size = make_tuple(self.params['kernel_size'])
        img = data.data['image']
        if self.params['type'] == 'box':
            cv2.boxFilter(src=img, dst=img, ddepth=0, ksize=kernel_size)
        if self.params['type'] == 'gaussian':
            cv2.GaussianBlur(src=img, dst=img, ksize=kernel_size, sigmaX=0)
        if self.params['type'] == 'median':
            cv2.medianBlur(src=img, dst=img, ksize=kernel_size)
        if self.params['type'] == 'bilateral':
            d = self.params['d']
            sigma_color = self.params['sigma_color']
            sigma_space = self.params['sigma_space']
            cv2.bilateralFilter(src=img, dst=img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        data.data['image'] = img


class Sharpen(Augmentation):

    def process(self, data: Data):
        img = data.data['image']
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
        data.data['image'] = img
