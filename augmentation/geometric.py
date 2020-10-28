import cv2
import numpy as np

from augmentation.data_types import Augmentation, Data


class Translation(Augmentation):
    def __init__(self, params):
        self.params = params

    def process(self, data: Data):
        (h, w) = data.data['image'].shape[:2]

        T = np.float32([[1, 0, self.params['t_y']], [0, 1, self.params['t_x']]])

        data.data['image'] = cv2.warpAffine(data.data['image'], T, (w, h))


class Scaling(Augmentation):
    def __init__(self, params):
        self.params = params

    def process(self, data: Data):
        (h, w) = data.data['image'].shape[:2]

        S = np.float32([[self.params['s_x'], 0, 0], [0, self.params['s_y'], 0]])

        data.data['image'] = cv2.warpAffine(data.data['image'], S, (w, h))


class Shear(Augmentation):
    def __init__(self, params):
        self.params = params

    def process(self, data: Data):
        (h, w) = data.data['image'].shape[:2]

        SH = np.float32([[1, self.params['sh_x'], 0], [self.params['sh_y'], 1, 0]])

        data.data['image'] = cv2.warpAffine(data.data['image'], SH, (w, h))


class Rotation(Augmentation):
    def __init__(self, params):
        self.params = params

    def process(self, data: Data):
        (h, w) = data.data['image'].shape[:2]
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
        if self.params['axis'] == 'horizontal':
            data.data['image'] = np.flipud(data.data['image'])

        if self.params['axis'] == 'vertical':
            data.data['image'] = np.fliplr(data.data['image'])
