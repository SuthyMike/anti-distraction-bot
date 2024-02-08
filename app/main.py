import argparse
import logging

import cv2

from app.capture.webcam_capture import WebcamCapture
from app.detection.focus_presence_detector import FocusPresenceDetector
from app.detection.person_presence_detector import PersonPresenceDector
from app.detection.yawn_presence_detector import YawnPresenceDetector
from app.utils.draw_facial_landmarks import draw_facial_landmarks
from app.utils.get_facial_landmarks import get_facial_landmarks

_FRAME_SKIP = 2


def main() -> None:
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format=("%(asctime)s" " %(levelname)s:" " %(message)s [in %(pathname)s:" "%(lineno)d]"),
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Anti-Distraction Bot")
    parser.add_argument("--show-feed", action="store_true", help="Show the video feed")
    args = parser.parse_args()

    webcam = WebcamCapture(logger)
    person_presence_detector = PersonPresenceDector(logger)
    focus_presence_detector = FocusPresenceDetector()
    yawn_presence_detector = YawnPresenceDetector(logger)

    try:
        frame_count = 0
        cached_landmarks = None
        while True:
            frame = webcam.get_frame()
            if frame is not None:
                print("GOT FRAME")
                if frame_count % _FRAME_SKIP == 0:
                    cached_landmarks = get_facial_landmarks(frame)
                frame_count += 1
                print(f"LAWMINGS {cached_landmarks}")
                if cached_landmarks is not None:
                    print("GOT LAWNMOWER")
                    draw_facial_landmarks(frame, cached_landmarks)
                    person_presence_detector.handle_presence_state(frame, cached_landmarks)
                    focus_presence_detector.handle_presence_state(frame, cached_landmarks)
                    yawn_presence_detector.handle_presence_state(frame, cached_landmarks)
                if args.show_feed:
                    print("FEED THEM")
                    cv2.imshow("Frame", frame)
                    cv2.waitKey(1)

    except Exception as e:
        logging.info("Error occurred", exc_info=e)
        raise
    finally:
        webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
