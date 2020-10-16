import cv2
import numpy as np

from augmentation.data_types import Augmentation, Data


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
