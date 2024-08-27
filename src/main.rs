mod audio;
mod capture;

use crate::audio::audio_player::play_audio;
use crate::capture::webcam_capture::WebcamCapture;
use anyhow::Result;

fn main() -> Result<()> {
    let mut webcam = WebcamCapture::new("Webcam Capture")?;

    if let Err(e) = play_audio("src/audio/assets/welcome.mp3") {
        eprintln!("Error playing audio: {}", e);
    }

    // show webcam frames
    webcam.show_frames()?;

    // Release the webcam
    webcam.release()?;

    Ok(())
}
