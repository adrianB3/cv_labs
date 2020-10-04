import os

import cv2
import numpy as np

from augmentation.data_types import Augmentation, Data
from ast import literal_eval as make_tuple
from augmentation.point_proc import *


class WriteImage(Augmentation):

    def process(self, data: Data):
        path = data.data['file_path']
        img = data.data['image']

        image_name = os.path.basename(path).split(".")[0]
        for aug_name in data.data['applied_augmentations']:
            image_name += '_' + aug_name
        image_name += "_" + str(data.data['count'])
        new_path = os.path.join(data.data['output_dir'], image_name + ".jpg")

        cv2.imwrite(filename=new_path, img=img)


class AddText(Augmentation):
    def __init__(self, params):
        self.params = params

    def process(self, data: Data):
        cv2.putText(data.data['image'], self.params['text'], make_tuple(self.params['position']), cv2.QT_FONT_NORMAL, 1,
                    (255, 0, 0), 1)


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
