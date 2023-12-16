from anti_distraction_bot.app.capture.webcam_capture import WebcamCapture
from anti_distraction_bot.app.detection.detect_presence import PresenceDetector
import cv2


def main():
    audio_paths = {
        "quit": "/Users/michael.sutherland/ai projects/anti-distraction-bot/anti_distraction_bot/assets/audio/quit.wav",
        "welcome_back": "/Users/michael.sutherland/ai projects/anti-distraction-bot/anti_distraction_bot/assets/audio/welcomeback.wav",
        "focus": "/Users/michael.sutherland/ai projects/anti-distraction-bot/anti_distraction_bot/assets/audio/focus.wav"
    }

    cascade_paths = {
        "frontal_face": "/Users/michael.sutherland/ai projects/anti-distraction-bot/anti_distraction_bot/assets/training data/haarcascade_frontalface_default.xml",
    }

    webcam = WebcamCapture()
    presence_detector = PresenceDetector(cascade_paths, audio_paths)
    try:
        while True:
            frame = webcam.get_frame()
            presence_detector.detect_presence(frame)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
