use rodio::{Decoder, OutputStream, Sink};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::error::Error;

pub fn play_audio(file_path: &str) -> Result<(), Box<dyn Error>> {
    let (_stream, handle) = OutputStream::try_default()?;

    let sink = Sink::try_new(&handle)?;

    let file = File::open(Path::new(file_path))?;
    let source = Decoder::new(BufReader::new(file))?;
    sink.append(source);

    sink.sleep_until_end();

    Ok(())
}
