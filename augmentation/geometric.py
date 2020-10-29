import cv2
import numpy as np

from augmentation.data_types import Augmentation, Data
from ast import literal_eval as make_tuple


class Translation(Augmentation):
    def __init__(self, params):
        self.params = params

    def process(self, data: Data):
        h, w, c = data.data['image'].shape
        prob = np.random.rand()
        if self.params['probability'] < prob:
            pass
        else:
            t_x_min, t_x_max = make_tuple(self.params['t_y_range'])
            t_y_min, t_y_max = make_tuple(self.params['t_x_range'])
            if t_y_min <= t_y_max and t_x_min <= t_x_max:
                t_y = np.random.randint(t_y_min, t_y_max)
                t_x = np.random.randint(t_x_min, t_x_max)
                T = np.float32([[1, 0, t_x], [0, 1, t_y]])

                data.data['image'] = cv2.warpAffine(data.data['image'], T, (w, h)).astype('uint8')


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
            scale_x_min, scale_x_max = make_tuple(self.params['scale_factor_x_range'])
            scale_y_min, scale_y_max = make_tuple(self.params['scale_factor_y_range'])
            if scale_x_min <= scale_x_max and scale_y_min <= scale_y_max:
                scale_x = np.random.uniform(scale_x_min, scale_x_max)
                scale_y = np.random.uniform(scale_y_min, scale_y_max)
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
            shear_f_min, shear_f_max = make_tuple(self.params['shear_factor_range'])
            if shear_f_min <= shear_f_max:
                rand_shear_factor = np.random.uniform(shear_f_min, shear_f_max)
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

        prob = np.random.rand()
        if self.params['probability'] < prob:
            pass
        else:
            min_angle, max_angle = make_tuple(self.params['angle_range'])
            if min_angle <= max_angle:
                angle = np.random.randint(min_angle, max_angle)
                M = cv2.getRotationMatrix2D((cx, cy), -angle,  1.0)
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])

                nw = int((h * sin) + (w * cos))
                nh = int((h * cos) + (w * sin))

                M[0, 2] += (nw / 2) - cx
                M[1, 2] += (nh / 2) - cy

                img = cv2.warpAffine(data.data['image'], M, (nw, nh))

                data.data['image'] = img


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
