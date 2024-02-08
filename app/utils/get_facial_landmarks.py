import cv2
import dlib  # type: ignore

from app.utils.draw_facial_landmarks import draw_facial_landmarks

# Initialize face detector and landmark predictor
_DETECTOR = dlib.get_frontal_face_detector()
_PREDICTOR = dlib.shape_predictor("assets/training/shape_predictor_68_face_landmarks.dat")


def get_facial_landmarks(frame: cv2.Mat) -> dlib.full_object_detection | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _DETECTOR(gray)
    print(f"faeces {faces}")

    for face in faces:
        landmarks = _PREDICTOR(gray, face)
        return landmarks

    return None
