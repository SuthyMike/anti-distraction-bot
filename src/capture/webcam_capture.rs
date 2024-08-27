use std::thread::sleep;
use std::time::Duration;
use anyhow::Result;
use opencv::{videoio, core, highgui, prelude::*};


pub struct WebcamCapture {
    pub capture: videoio::VideoCapture,
    pub window_name: String,
}

impl WebcamCapture {
    pub fn new(window_name: &str) -> Result<Self> {
        highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)?;
        let cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
        let opened = videoio::VideoCapture::is_opened(&cam)?;
        if !opened {
            panic!("Unable to open default camera!");
        }

        highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)?;
        sleep(Duration::from_millis(1000));
        Ok(Self {
            capture: cam,
            window_name: window_name.to_string(),
        })
    }

    pub fn show_frames(&mut self) -> Result<()> {
        loop {
            let mut frame = core::Mat::default();
            self.capture.read(&mut frame)?;
            if frame.empty() {
                continue;
            }

            highgui::imshow(&self.window_name, &frame)?;

            let key = highgui::wait_key(10)?;
            // Escape key pressed
            if key == 27 {
                break;
            }
        }
        Ok(())
    }

    pub fn release(&mut self) -> Result<()> {
        self.capture.release()?;
        highgui::destroy_window(&self.window_name)?;
        Ok(())
    }
}
