// facial_landmarks.rs

use opencv::prelude::*;
use opencv::core::{Rect, Vector, Point, Ptr, Mat, Size};
use opencv::imgproc;
use opencv::objdetect::CascadeClassifier;
use anyhow::Result;
use opencv::face::{FacemarkLBF, FacemarkLBF_Params};


pub struct FacialLandmarkDetector {
    face_cascade: CascadeClassifier,
    facemark: Ptr<FacemarkLBF>,
}

impl FacialLandmarkDetector {
    pub fn new() -> Result<Self> {
        // load frontal face detector
        let face_cascade = CascadeClassifier::new("/Users/michael.sutherland/Projects/anti-distraction bot/src/assets/training/haarcascade_frontalface_default.xml")?;

        // Use default parameters
        let params = FacemarkLBF_Params::default()?;
        let mut facemark = FacemarkLBF::create(&params)?;
        // load pre-trained model
        facemark.load_model("/Users/michael.sutherland/Projects/anti-distraction bot/src/assets/training/lbfmodel.yaml")?;

        Ok(Self { face_cascade, facemark })
    }

    pub fn detect_faces(&mut self, frame: &Mat) -> Result<Vector<Rect>> {
        // Convert to grayscale
        let mut gray = Mat::default();
        imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

        // Detect faces
        let mut faces = Vector::<Rect>::new();
        self.face_cascade.detect_multi_scale(
            &gray,
            &mut faces,
            1.05,
            5,
            0,
            Size::new(30, 30),
            Size::new(0, 0),
        )?;

        Ok(faces)
    }

    pub fn get_facial_landmarks(&mut self, frame: &Mat) -> Result<Option<Vector<Point>>> {
        // Detect face in the frame
        let faces = self.detect_faces(frame)?;

        if faces.len() > 0 {
            println!("faces: {}", faces.len());
            let mut landmarks = Vector::<Vector<Point>>::new();
            let _ = self.facemark.fit(&frame, &faces, &mut landmarks);

            match landmarks.get(0) {
                Ok(face_landmarks) => {
                    println!("landmarks in get_facial_landmarks: {}", face_landmarks.len());
                    return {
                        eprintln!("landmarks in get_facial_landmarks: {}", face_landmarks.len());
                        Ok(Some(face_landmarks.clone()))
                    };
                }
                Err(_) => return Ok(None), // TODO handle no landmarks found
            }
        }

        Ok(None)
    }
}
