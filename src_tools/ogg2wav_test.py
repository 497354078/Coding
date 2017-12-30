# -*- coding:utf-8 -*-
import pydub
import numpy as np
from pydub import AudioSegment ###需要安装pydub、ffmpeg
import wave
import io
#先从本地获取mp3的bytestring作为数据样本
fp=open("/Users/ljpc/Desktop/audio/5-170338-A.ogg",'rb') #7061-6-0-0.wav
data=fp.read()
fp.close()
#主要部分
aud=io.BytesIO(data)
sound=AudioSegment.from_file(aud,format='ogg').split_to_mono()[0]
raw_data = sound.raw_data
print raw_data.__len__(), type(raw_data)
print sound.sample_width
print sound.channels
print sound.frame_rate
print sound.frame_width
print sound.frame_count()
#写入到文件，验证结果是否正确。
l=len(raw_data)
f=wave.open("/Users/ljpc/Desktop/audio/5-170338-A.wav",'wb')
f.setnchannels(1)
f.setsampwidth(sound.sample_width)
f.setframerate(sound.frame_rate/2)
#f.setnframes(raw_data.__len__())
f.writeframes(raw_data)
f.close()

def _load_audio(path, duration, sr):
    audio = pydub.AudioSegment.silent(duration=duration)
    audio = audio.overlay(
        pydub.AudioSegment.from_file(path).set_frame_rate(sr).split_to_mono()[0]
                        )[0:duration]
    raw = np.fromstring(audio._data, dtype="int16")*1.0/0x7FFF
    #raw = (np.fromstring(audio._data, dtype="int16") + 0.5) / (0x7FFF + 0.5)   # convert to float
    return raw, sr

y, sr = _load_audio('/Users/ljpc/Desktop/audio/7061-6-0-0.wav', 2250, 22050)
print 'y: ', y.shape, y.min(), y.max()
print 'sr: ', sr

import sys
sys.path.append('../')
from util import plot_wave
import librosa
import librosa.display
plot_wave(y, sr)


