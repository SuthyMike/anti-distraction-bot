import logging
import time

import cv2

_SKIP_FRAMES = 2


class WebcamCapture:
    def __init__(self, logger: logging.Logger):
        self.cap = cv2.VideoCapture(0)
        self.logger = logger

        if not self.cap.isOpened():
            raise OSError("Failed to initialise webcam")
        time.sleep(2)  # wait for webcam to initialise and adjust light levels

    def get_frame(self) -> cv2.Mat | None:
        # skip some frames to reduce frame rate
        for _ in range(_SKIP_FRAMES):
            self.cap.read()
        ret, frame = self.cap.read()
        if not ret:
            self.logger.info("Failed to capture webcam")
            return None
        return frame

    def release(self) -> None:
        self.cap.release()
