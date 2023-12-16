import cv2
import time


class WebcamCapture:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Failed to initialise webcam")
        time.sleep(2)  # wait for webcam to initialise and adjust light levels

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture webcam")
            return None
        return frame

    def release(self):
        self.cap.release()
