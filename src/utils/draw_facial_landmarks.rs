use opencv::core::{Point, Mat, Scalar, Vector};
use opencv::imgproc::{self, FONT_HERSHEY_SIMPLEX, LINE_8};
use anyhow::Result;

pub fn draw_facial_landmarks(frame: &mut Mat, landmarks: &Vector<Point>) -> Result<()> {
    // Facial landmark points we care about (using indices)
    let facial_points = [
        (30, "Nose tip"),
        (8, "Chin"),
        (36, "Left eye left corner"),
        (45, "Right eye right corner"),
        (48, "Left mouth corner"),
        (54, "Right mouth corner"),
    ];

    // Draw circles and labels at key facial points
    for &(index, label) in &facial_points {
        if let Ok(part) = landmarks.get(index as usize) {
            let x = part.x;
            let y = part.y;

            imgproc::circle(frame, Point::new(x, y), 4, Scalar::new(0.0, 255.0, 0.0, 0.0), -1, LINE_8, 0)?;
            imgproc::put_text(frame, label, Point::new(x - 10, y - 10), FONT_HERSHEY_SIMPLEX, 0.4, Scalar::new(255.0, 255.0, 255.0, 0.0), 1, LINE_8, false)?;
        }
    }

    Ok(())
}
