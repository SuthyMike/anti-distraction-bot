import logging
import time

import cv2
import dlib

from app.audio.player import play_audio
from app.detection.base_presence_detector import BasePresenceDetector

_NOT_PRESENT_THRESHOLD = 3


class PersonPresenceDector(BasePresenceDetector):
    def __init__(self, logger: logging.Logger):
        super().__init__()
        self.was_present = False
        self.quit_audio_path = "assets/audio/quit.wav"
        self.welcome_back_audio_path = "assets/audio/welcomeback.wav"
        self.no_presence_timer_start: float | None = None
        self.logger = logger

    def handle_presence_state(self, _frame: cv2.Mat, landmarks: dlib.full_object_detection) -> None:
        if landmarks:
            if self.was_present:
                self.logger.info("Playing welcome back audio...")
                play_audio(self.welcome_back_audio_path)
                self.was_present = False
            self.no_presence_timer_start = None
        elif self.no_presence_timer_start is None:
            self.logger.info("Starting timer for absence detection...")
            self.no_presence_timer_start = time.time()
        elif time.time() - self.no_presence_timer_start > _NOT_PRESENT_THRESHOLD \
                and not self.was_present:
            self.was_present = True
            self.logger.info("Playing quit audio...")
            play_audio(self.quit_audio_path)
