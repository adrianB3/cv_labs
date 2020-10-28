import cv2
import numpy as np

from augmentation.data_types import Augmentation, Data


class Translation(Augmentation):
    def __init__(self, params):
        self.params = params

    def process(self, data: Data):
        h, w, c = data.data['image'].shape

        T = np.float32([[1, 0, self.params['t_y']], [0, 1, self.params['t_x']]])

        data.data['image'] = cv2.warpAffine(data.data['image'], T, (w, h))


class Scaling(Augmentation):
    def __init__(self, params):
        self.params = params

    def process(self, data: Data):
        img = data.data['image']
        h, w, c = img.shape

        prob = np.random.rand()
        if self.params['probability'] < prob:
            pass
        else:
            scale_x = np.random.uniform(self.params['scale_factor_x'])
            scale_y = np.random.uniform(self.params['scale_factor_y'])
            resize_scale_x = 1 + scale_x
            resize_scale_y = 1 + scale_y

            img = cv2.resize(img, None, fx=resize_scale_x, fy=resize_scale_y)

            data.data['image'] = img


class Shear(Augmentation):
    def __init__(self, params):
        self.params = params

    def process(self, data: Data):
        img = data.data['image']
        h, w, c = img.shape

        prob = np.random.rand()
        if self.params['probability'] < prob:
            pass
        else:
            rand_shear_factor = np.random.uniform(self.params['shear_factor'])
            SH = np.float32([[1, abs(rand_shear_factor), 0], [0, 1, 0]])
            nW = int(img.shape[1] + abs(rand_shear_factor * img.shape[0]))
            img = cv2.warpAffine(img, SH, (nW, h)).astype('uint8')
            img = cv2.resize(img, (w, h))

            data.data['image'] = img


class Rotation(Augmentation):
    def __init__(self, params):
        self.params = params

    def process(self, data: Data):
        (h, w, c) = data.data['image'].shape
        (cx, cy) = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D((cx, cy), -self.params['angle'], 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nw = int((h * sin) + (w * cos))
        nh = int((h * cos) + (w * sin))

        M[0, 2] += (nw / 2) - cx
        M[1, 2] += (nh / 2) - cy

        data.data['image'] = cv2.warpAffine(data.data['image'], M, (nw, nh))


class Flip(Augmentation):
    def __init__(self, params):
        self.params = params

    def process(self, data: Data):

        prob = np.random.rand()
        if self.params['probability'] < prob:
            pass
        else:
            if self.params['axis'] == 'horizontal':
                data.data['image'] = cv2.flip(data.data['image'], 1)

            if self.params['axis'] == 'vertical':
                data.data['image'] = cv2.flip(data.data['image'], 0)

            if self.params['axis'] == 'both':
                data.data['image'] = cv2.flip(data.data['image'], -1)
