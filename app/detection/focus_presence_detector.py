import time
from typing import Any

import cv2
import dlib
import numpy as np
from numpy import ndarray

from app.audio.player import play_audio
from app.detection.base_presence_detector import BasePresenceDetector

_YAW_THRESHOLD = 30
_TIMER_THRESHOLD = 3

class FocusPresenceDetector(BasePresenceDetector):
    def __init__(self) -> None:
        super().__init__()
        self.focus_audio = "assets/audio/focus.wav"
        self.timer_start: float | None = None
        self.yaw_out_of_range = False

    def handle_presence_state(self, frame: cv2.Mat, landmarks: dlib.full_object_detection) -> None:
        if not landmarks:
            return

        # Extract relevant landmark points
        image_points = np.array(
            [
                (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
                (landmarks.part(8).x, landmarks.part(8).y),  # Chin
                (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
                (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corner
                (landmarks.part(48).x, landmarks.part(48).y),  # Left Mouth corner
                (landmarks.part(54).x, landmarks.part(54).y),  # Right mouth corner
            ],
            dtype="double",
        )

        # Tweak camera settings based on resolution e.g. 1920 x 1080
        size = frame.shape
        print(f"SIZE {size}")
        focal_length = size[1]
        print(f"FOCAL LEN {focal_length}")
        center = (size[1] / 2, size[0] / 2)
        print(f"CENTER {center}")
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double"
        )

        # Assuming no lens distortion
        dist_coeffs = np.zeros((4, 1))

        # SolvePnP to get rotation vector
        # https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.generic_face,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        # Convert the rotation vector to Euler angles
        euler_angles = _rotation_vector_to_euler_angles(rotation_vector)

        # Determine if the user is facing the camera or turned away
        yaw = euler_angles[1]
        if yaw < -_YAW_THRESHOLD or yaw > _YAW_THRESHOLD:
            # Start the timer if it's not already started
            if self.timer_start is None:
                self.timer_start = time.time()
            elif time.time() - self.timer_start > _TIMER_THRESHOLD:
                # If yaw has been out of range for more than 3 seconds, play audio
                play_audio(self.focus_audio)
                self.timer_start = None  # Reset timer after playing audio
        else:
            # Reset timer if yaw returns to the -30 to 30 range
            self.timer_start = None

        if yaw < -_YAW_THRESHOLD:
            direction = "Looking Left"
        elif yaw > _YAW_THRESHOLD:
            direction = "Looking Right"
        else:
            direction = "Facing Forward"

        cv2.putText(frame, direction, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


# Function to convert rotation vector to Euler angles
# function from https://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
# the gimbal lock case does not seem to be a problem in this simplified solution so the two-solution scenario has been omitted
def _rotation_vector_to_euler_angles(rotation_vector) -> ndarray[Any, Any]:
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(
        rotation_matrix[0, 0] * rotation_matrix[0, 0]
        + rotation_matrix[1, 0] * rotation_matrix[1, 0]
    )
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0

    return np.degrees(np.array([x, y, z]))
