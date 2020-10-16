import cv2

from augmentation.data_types import Augmentation, Data
from ast import literal_eval as make_tuple


class Blur(Augmentation):
    def __init__(self, params):
        self.params = params

    def process(self, data: Data):
        kernel_size = make_tuple(self.params['kernel_size'])
        img = data.data['image']
        if self.params['type'] == 'box':
            img = cv2.boxFilter(src=img, ddepth=0, ksize=kernel_size)
        if self.params['type'] == 'gaussian':
            img = cv2.GaussianBlur(img, ksize=kernel_size, sigmaX=0)
        if self.params['type'] == 'median':
            img = cv2.medianBlur(img, ksize=kernel_size)
        if self.params['type'] == 'bilateral':
            d = self.params['d']
            sigma_color = self.params['sigma_color']
            sigma_space = self.params['sigma_space']
            img = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
        data.data['image'] = img
