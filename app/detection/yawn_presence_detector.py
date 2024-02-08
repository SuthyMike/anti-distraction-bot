import logging

import cv2
import dlib
import numpy as np

from app.audio.player import play_audio
from app.detection.base_presence_detector import BasePresenceDetector


class YawnPresenceDetector(BasePresenceDetector):
    def __init__(self, logger: logging.Logger) -> None:
        super().__init__()
        self.wake_up_audio = "assets/audio/wake_up.wav"
        self.yawn_threshold = 0.8
        self.logger = logger

    def handle_presence_state(self, _frame: cv2.Mat, landmarks: dlib.full_object_detection) -> None:
        if not landmarks:
            return

        # Extract mouth points
        mouth_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])

        # Vertical distance between two middle points of the mouth
        vertical_dist = np.linalg.norm(mouth_points[3] - mouth_points[9])

        # Horizontal distance between two side points of the mouth
        horizontal_dist = np.linalg.norm(mouth_points[0] - mouth_points[6])

        # Calculate Mouth Aspect Ratio
        mar = vertical_dist / horizontal_dist

        if mar > self.yawn_threshold:
            self.logger.info("User is yawning, playing wake up audio")
            play_audio(self.wake_up_audio)
