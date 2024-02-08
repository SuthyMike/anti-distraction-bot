from abc import ABC, abstractmethod

import cv2
import dlib
import numpy as np


class BasePresenceDetector(ABC):
    def __init__(self) -> None:
        # Create a 3D model of a generic face
        self.generic_face = np.array(
            [
                (0.0, 0.0, 0.0),  # Nose tip
                (0.0, -330.0, -65.0),  # Chin
                (-225.0, 170.0, -135.0),  # Left eye left corner
                (225.0, 170.0, -135.0),  # Right eye right corner
                (-150.0, -150.0, -125.0),  # Left Mouth corner
                (150.0, -150.0, -125.0),  # Right mouth corner
            ]
        )

    @abstractmethod
    def handle_presence_state(
            self, frame: cv2.Mat, landmarks: dlib.full_object_detection
    ) -> None:
        """Handles the detection of presence for a particular behaviour or object
        This method should be implemented by all subclasses.
        """
