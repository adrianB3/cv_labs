import cv2

from augmentation.data_types import Augmentation, Data


class AddMyName(Augmentation):
    def process(self, data: Data):
        cv2.putText(data.data['image'], "Adrian", (50, 50), cv2.QT_FONT_NORMAL, 1,
                    (255, 0, 0), 1)
