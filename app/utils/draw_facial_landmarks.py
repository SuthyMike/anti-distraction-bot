import cv2
import dlib


def draw_facial_landmarks(frame: cv2.Mat, landmarks: dlib.full_object_detection) -> None:
    # Define facial landmark points to draw and label
    facial_points = {
        30: "Nose tip",
        8: "Chin",
        36: "Left eye left corner",
        45: "Right eye right corner",
        48: "Left mouth corner",
        54: "Right mouth corner",
    }

    # Draw circles and labels at key facial points
    for point, label in facial_points.items():
        x = landmarks.part(point).x
        y = landmarks.part(point).y
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        cv2.putText(
            frame, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
        )
