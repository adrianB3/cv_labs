import cv2
import numpy as np


class BlankSeqDetector:
    def __init__(self):
        self.backSub = cv2.createBackgroundSubtractorMOG2(4, 30, detectShadows=False)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.thresh = 0.14

    def process(self, frame_buffer):
        blured = cv2.blur(src=frame_buffer[0], ksize=(15, 15))
        cv2.imshow("Blurred", blured)
        mask = self.backSub.apply(blured)
        fgmask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        wh_count = cv2.countNonZero(fgmask)
        prc_mask = (wh_count / (frame_buffer[0].shape[0] * frame_buffer[0].shape[1])) * 100
        print(prc_mask)
        # contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(frame_buffer[0], contours, -1, (0, 255, 0), 3)
        mask_with_img = cv2.bitwise_and(frame_buffer[0], frame_buffer[0], mask=fgmask)
        cv2.imshow("Foreground Mask", mask_with_img)
        return prc_mask < self.thresh


class BlankSeqDetectorOF:
    def __init__(self):
        pass

    def process(self, frame_buff):
        frame1 = frame_buff[0]
        frame2 = frame_buff[1]
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("Optical Flow", bgr)
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
