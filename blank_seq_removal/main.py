import cv2
import os
import yaml

from collections import deque

from blank_seq_removal.blank_seq_remove import BlankSeqDetector

cfg_path = open('cfg.yml')
cfg = yaml.load(cfg_path, Loader=yaml.FullLoader)
step = 20


def main_loop():
    if cfg['use_camera']:
        cap = cv2.VideoCapture(0)
    else:
        if os.path.exists(cfg['video_path']):
            cap = cv2.VideoCapture(cfg['video_path'])
        else:
            raise FileNotFoundError("Video file doesn't exist.")

    bsd = BlankSeqDetector()
    buffer = deque(maxlen=cfg['buffer_size'])
    is_static = False

    while True:
        ret, frame = cap.read()
        h, w, c = frame.shape

        buffer.append(frame)

        if len(buffer) == cfg['buffer_size']:
            is_static = bsd.process(buffer)
            buffer.pop()

        if is_static:
            for i in range(0, h*step, step):
                cv2.line(frame, (i, 0), (i, w), (0, 0, 255), 2)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_loop()
