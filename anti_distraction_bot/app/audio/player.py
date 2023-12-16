import simpleaudio as sa


def play_audio(file_path):
    audio = sa.WaveObject.from_wave_file(file_path)
    return audio.play()

