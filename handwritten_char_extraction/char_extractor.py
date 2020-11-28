import cv2
import numpy as np


class CharExtractor:
    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def process(self, image):
        h, w, c = image.shape
        img = image.copy()
        if c == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        closed = self.preprocess(img)

        contours = self.extract_lines2(closed)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return image

    def preprocess(self, img):
        ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        clean_img = cv2.medianBlur(th, 5)
        closed = cv2.morphologyEx(clean_img, cv2.MORPH_CLOSE, self.kernel)
        return closed

    def extract_lines1(self, img):
        h, w = img.shape
        pts = cv2.findNonZero(img)
        ret = cv2.minAreaRect(pts)
        (cx, cy), (r_w, r_h), ang = ret
        M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        hist = cv2.reduce(rotated, 1, cv2.REDUCE_AVG).reshape(-1)

        th = 2
        uppers = [y for y in range(h - 1) if hist[y] <= th < hist[y + 1]]
        lowers = [y for y in range(h - 1) if hist[y] > th >= hist[y + 1]]

        rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
        for y in uppers:
            cv2.line(rotated, (0, y), (w, y), (255, 0, 0), 1)

        for y in lowers:
            cv2.line(rotated, (0, y), (w, y), (0, 255, 0), 1)
        return rotated

    def extract_lines2(self, img):
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
        dilation = cv2.dilate(img, rect_kernel, iterations=1)
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours
