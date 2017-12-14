import os
import sys
sys.path.append('/home/lj/tools')
import numpy as np
import cPickle
import librosa
from python_speech_features import logfbank
from util import *

if __name__ == '__main__':

    sr = 22050
    audioPath = ''
    feaPath = 'feature/feature_UrbanSound8K/{:s}/logfbank'.format(str(sr))

    check_path(audioPath)
    make_path(feaPath)

    cnt = 0
    for root, dirs, filenames in os.walk(audioPath):
        for wavFile in filenames:
            if not wavFile.endswith('.ogg'):
                continue
            feaFile = os.path.join(feaPath, wavFile.replace('.ogg', '.fea'))
            if os.path.isfile(feaFile):
                continue
            y, sr = librosa.load(os.path.join(root, wavFile), sr=sr, mono=True)
            feaMatrix = logfbank(y, samplerate=sr,
                                 winlen=0.025, winstep=0.01,
                                 nfilt=40, nfft=int(sr*0.025))
            cPickle.dump(feaMatrix, open(feaFile, 'wb'),
                         protocol=cPickle.HIGHEST_PROTOCOL)
            cnt += 1
            print 'cnt: {:d} wavFile: {:s} feaMatrix: {:s}'.format(
                cnt, wavFile, feaMatrix.shape)
            exit(0)

