import cv2


class BlankSeqDetector:
    def __init__(self):
        self.backSub = cv2.createBackgroundSubtractorMOG2(2, 100, detectShadows=False)

    def process(self, frame_buffer):
        blured = cv2.GaussianBlur(src=frame_buffer[0], ksize=(15, 15), sigmaX=0)
        mask = self.backSub.apply(blured)
        cv2.imshow("Foreground Mask", mask)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame_buffer[0], contours, -1, (0, 255, 0), 3)

        # nb = cv2.countNonZero(mask)
        if len(contours) < 3:
            return True
        return False
