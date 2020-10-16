import os

import cv2

from ast import literal_eval as make_tuple
from augmentation.point_proc import *
from augmentation.geometric import *


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
