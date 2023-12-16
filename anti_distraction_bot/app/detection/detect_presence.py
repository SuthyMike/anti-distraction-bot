import cv2
import time
from ..audio.player import play_audio  # Update the import path as needed


class PresenceDetector:
    def __init__(self, cascade_paths, audio_paths):
        self.frontal_face_cascade = cv2.CascadeClassifier(cascade_paths['frontal_face'])
        self.focus_audio_path = audio_paths['focus']
        self.quit_audio_path = audio_paths['quit']
        self.welcome_back_audio_path = audio_paths['welcome_back']
        self.min_face_size = (200, 200)
        self.max_face_size = (600, 600)
        self.no_presence_timer_start = None
        self.was_present = False

    def detect_presence(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        frontal_faces = self.frontal_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        frontal_present = self._is_face_detected(frontal_faces, frame)  # remove frame when removing imshow

        self._handle_presence_state(frontal_present)
        cv2.imshow('Human Detection', frame)
        cv2.waitKey(1)

    def _is_face_detected(self, faces, frame):
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if self.min_face_size[0] <= w <= self.max_face_size[0] and self.min_face_size[1] <= h <= self.max_face_size[1]:
                return True
        return False

    def _handle_presence_state(self, present):
        if present:
            if self.was_present:
                print('Playing welcome back audio...')
                play_audio(self.welcome_back_audio_path).wait_done()
                self.was_present = False
            self.no_presence_timer_start = None
        else:
            if self.no_presence_timer_start is None:
                print('Starting timer for absence detection...')
                self.no_presence_timer_start = time.time()
            elif time.time() - self.no_presence_timer_start > 3 and not self.was_present:
                self.was_present = True
                print('Playing quit audio...')
                play_audio(self.quit_audio_path).wait_done()
