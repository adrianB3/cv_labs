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
            width = int(rect[1][0])
            height = int(rect[1][1])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (0, 0, 0), 2)
            letters, warped = self.extract_letters(clean, box, width, height)
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

    def extract_letters(self, img, box, width, height):

        src_pts = box.astype("float32")
        # coordinate of the points in box points after the rectangle has been
        # straightened
        dst_pts = np.array([[0, height - 1],
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1]], dtype="float32")

        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # directly warp the rotated rectangle to get the straightened rectangle
        warped = cv2.warpPerspective(img, M, (width, height))

        thinned = cv2.ximgproc.thinning(warped)

        x_sum = cv2.reduce(thinned, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)

        warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        limits = []
        consecutive_line_letters = []
        for i in range(0, len(x_sum[0])):
            if x_sum[0][i] == 0:  # potentsial segmentation columns
                limits.append(i)
                if len(consecutive_line_letters) != 0:
                    letter = warped[0:warped.shape[0], consecutive_line_letters[0]:consecutive_line_letters[-1]]
                    if len(letter) != 0:
                        cv2.imwrite(os.path.join(os.getcwd(), "characters", "img" + str(i) + ".png"), letter)
                    consecutive_line_letters.clear()
            else:
                consecutive_line_letters.append(i)
            # if x_sum[0][i] == 255:
            #     cv2.line(warped, (i, 0), (i, warped.shape[1]), (0, 255, 0), 1)
        return limits, warped
