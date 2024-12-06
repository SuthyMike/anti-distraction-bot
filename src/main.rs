mod audio;
mod capture;
mod utils;
mod detection;

use crate::audio::audio_player::play_audio;
use crate::capture::webcam_capture::WebcamCapture;
use crate::detection::get_facial_landmarks::FacialLandmarkDetector;
use anyhow::Result;
use crate::utils::draw_facial_landmarks::draw_facial_landmarks;
use opencv::{highgui, prelude::*};

fn main() -> Result<()> {
    // initialise new webcam feed
    let mut webcam = WebcamCapture::new("Webcam Capture")?;

    if let Err(e) = play_audio("/Users/michael.sutherland/Projects/anti-distraction bot/src/assets/audio/welcome.mp3") {
        eprintln!("Error playing audio: {}", e);
    }

    let mut facial_landmark_detector = FacialLandmarkDetector::new()?;

    loop {
        let mut frame = webcam.get_frame()?;

        if frame.empty() {
            println!("frame is empty");
            continue;
        }

        // detect facial features
        if let Ok(Some(landmarks)) = facial_landmark_detector.get_facial_landmarks(&frame) {
            println!("Landmarks detected: {:?}", landmarks);
            // Draw the detected landmarks on the frame
            draw_facial_landmarks(&mut frame, &landmarks)?;
        }

        // Display the frame with drawn landmarks
        highgui::imshow("Webcam Capture", &frame)?;
        if highgui::wait_key(10)? == 27 {
            break;
        }
    }

    // Release the webcam
    webcam.release()?;

    Ok(())
}
