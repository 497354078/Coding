import sys
import numpy as np
import io
import wave
import pydub
from pydub import AudioSegment

def _load_audio(rawFile, duration=5000, sr=22050):
    audio = pydub.AudioSegment.silent(duration=duration)
    audio = audio.overlay(
        pydub.AudioSegment.from_file(rawFile).set_frame_rate(sr).split_to_mono()[0]
                        )[0:duration]
    y = np.fromstring(audio._data, dtype="int16")*1.0/0x7FFF
    #y = (np.fromstring(audio._data, dtype="int16") + 0.5) / (0x7FFF + 0.5)   # convert to float
    return audio._data, y, sr

def _save_audio(files, data, sr):
    f = wave.open(files, 'wb')
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sr)
    f.writeframes(data)
    f.close()

def ogg2wav(rawFile, destFile, duration=5000, sr=22050):
    data, y, sr = _load_audio(rawFile, duration=duration, sr=sr)
    _save_audio(destFile, data, sr)

if __name__ == '__main__':
    rawFile = '/Users/ljpc/Desktop/audio/1-54958-A.ogg'#'7061-6-0-0.wav'
    destFile = '/Users/ljpc/Desktop/audio/ogg2wav.wav'
    ogg2wav(rawFile, destFile, duration=5000, sr=16000)

