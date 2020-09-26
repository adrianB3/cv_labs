import os

import cv2

from augmentation.data_types import Augmentation, Data


class WriteImage(Augmentation):

    def process(self, data: Data):
        path = data.data['file_path']
        img = data.data['image']

        image_name = os.path.basename(path).split(".")[0]
        for aug_name in data.data['applied_augmentations']:
            image_name += '_' + aug_name
        image_name += "_" + str(data.data['count'])
        new_path = os.path.join(data.data['output_dir'], image_name + "_aug.jpg")

        cv2.imwrite(filename=new_path, img=img)

        data.data['applied_augmentations'].clear()
