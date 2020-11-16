import cv2
import numpy as np


class BlankSeqDetector:
    def __init__(self):
        self.backSub = cv2.createBackgroundSubtractorMOG2(10, 100, detectShadows=False)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    def process(self, frame_buffer):
        blured = cv2.GaussianBlur(src=frame_buffer[0], ksize=(15, 15), sigmaX=0)
        mask = self.backSub.apply(blured)

        fgmask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        cv2.imshow("Foreground Mask", fgmask)
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame_buffer[0], contours, -1, (0, 255, 0), 3)

        # nb = cv2.countNonZero(mask)
        if len(contours) < 3:
            return True
        return False


class BlankSeqDetectorRPCA:
    def __init__(self):
        pass

    def process(self, frame_buffer):
        pass

    def create_data_matrix_from_video(self, clip, k, scale):
        frames = []
        for i in range(k * int(clip.duration)):
            frame = clip.get_frame(i / float(k))
            frame = self.rgb2grey(frame).astype(int)
            frame = cv2.resize(frame, scale, scale).flatten()
            frames.append(frame)
        return np.vstack(frames).T  # stack images horizontally

    def rgb2grey(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
