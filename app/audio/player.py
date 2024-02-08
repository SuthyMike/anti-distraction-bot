import simpleaudio  # type: ignore


def play_audio(file_path: str) -> None:
    audio = simpleaudio.WaveObject.from_wave_file(file_path)
    audio.play().wait_done()
