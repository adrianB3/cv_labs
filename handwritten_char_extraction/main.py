from pathlib import Path

import cv2
import os
import yaml

from handwritten_char_extraction.char_extractor import CharExtractor

cfg_path = open('cfg.yml')
cfg = yaml.load(cfg_path, Loader=yaml.FullLoader)

ce = CharExtractor()

if __name__ == "__main__":
    cv2.namedWindow("Img", cv2.WINDOW_GUI_EXPANDED | cv2.WINDOW_FREERATIO)
    img_path = Path(cfg['test_img'])
    img_path = os.path.join(os.getcwd(), img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    out = ce.process(img)

    cv2.imshow("Img", out)

    cv2.waitKey()
    cv2.destroyAllWindows()
