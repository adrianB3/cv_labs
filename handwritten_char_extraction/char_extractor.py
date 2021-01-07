import cv2
import numpy as np
import os


class CharExtractor:
    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def process(self, image):
        img = image.copy()
        clean = self.preprocess(img)
        lines = self.extract_lines(clean)
        count = 0
        for line in lines:
            rect = cv2.minAreaRect(line)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cropped = self.crop_minAreaRect(clean, rect, box)
            if cropped.size != 0:
                letters, warped = self.extract_letters(cropped)
                cv2.drawContours(img, [box], 0, (0, 0, 0), 2)
                for limit in letters:
                    cv2.line(warped, (limit, 0), (limit, warped.shape[1]), (0, 255, 0), 1)
                cv2.imshow("letters" + str(count), warped)
                count += 1

        return img

    def preprocess(self, img):
        ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        clean_img = cv2.medianBlur(th, 5)
        closed = cv2.morphologyEx(clean_img, cv2.MORPH_CLOSE, self.kernel)
        clean_img_again = cv2.medianBlur(closed, 5)
        return clean_img_again

    def extract_lines(self, img):
        img2 = np.zeros(img.shape, dtype=np.uint8)
        img2.fill(0)

        minLineLength = 1
        maxLineGap = 100
        lines = cv2.HoughLinesP(img, 3, np.pi / 180, 850, minLineLength=minLineLength, maxLineGap=maxLineGap)
        for x in range(0, len(lines)):
            for x1, y1, x2, y2 in lines[x]:
                cv2.line(img2, (x1, y1), (x2, y2), (255, 0, 0), 10)

        clean = cv2.morphologyEx(img2, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))

        cv2.imshow("lines", clean)
        contours, hierarchy = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours

    def extract_letters(self, crop):

        thinned = cv2.ximgproc.thinning(crop)

        x_sum = cv2.reduce(thinned, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)

        warped = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        limits = []
        consecutive_line_letters = []
        for i in range(0, len(x_sum[0])):
            if x_sum[0][i] == 0:  # potentsial segmentation columns
                limits.append(i)
                if len(consecutive_line_letters) != 0:
                    letter = warped[0:warped.shape[0], consecutive_line_letters[0]:consecutive_line_letters[-1]]
                    # if len(letter) != 0:
                        # cv2.imwrite(os.path.join(os.getcwd(), "characters", "img" + str(i) + ".png"), letter)
                    consecutive_line_letters.clear()
            else:
                consecutive_line_letters.append(i)
            # if x_sum[0][i] == 255:
            #     cv2.line(warped, (i, 0), (i, warped.shape[1]), (0, 255, 0), 1)
        return limits, warped

    def crop_minAreaRect(self, img, rect, box):

        W = rect[1][0]
        H = rect[1][1]

        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)

        angle = rect[2]
        if angle < -45:
            angle += 90

        # Center of rectangle in source image
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        # Size of the upright rectangle bounding the rotated rectangle
        size = (x2 - x1, y2 - y1)
        M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)
        # Cropped upright rectangle
        cropped = cv2.getRectSubPix(img, size, center)
        cropped = cv2.warpAffine(cropped, M, size)
        croppedW = H if H > W else W
        croppedH = H if H < W else W
        # Final cropped & rotated rectangle
        croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW), int(croppedH)), (size[0] / 2, size[1] / 2))

        return croppedRotated
