import os
import sys
import copy
import numpy as np
import cPickle as pickle
import librosa
import librosa.display
import matplotlib.pyplot as plt
from util import *
from python_speech_features import fbank



def plot_figure(y, sr, mark, audioName, pltSave):

    plt.figure()

    plt.subplot(3, 1, 1)
    plt.xlim(0, len(mark))
    plt.ylim(-1, 1)
    librosa.display.waveplot(y, sr=sr)
    plt.title(audioName)

    plt.subplot(3, 1, 2)
    plt.xlim(0, len(mark))
    plt.ylim(0, 1)
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    librosa.display.specshow(D, x_axis='time', y_axis='log')
    plt.title('Log-frequency power spectrogram')

    plt.subplot(3, 1, 3)
    plt.xlim(0, len(mark))
    plt.ylim(0, 1)
    color = []
    for i in range(len(mark)):
        if mark[i] == False:
            color.append('lightgrey')
        elif mark[i] == True:
            color.append('lightcyan')
    plt.bar(range(len(mark)), np.ones(mark.shape), 1, color=color, align='edge')
    plt.xlabel("Time")
    plt.ylabel("silent-label")
    plt.title("alpha={:s}".format(str(alpha)))

    plt.tight_layout()
    plt.savefig(pltSave)
    #plt.show()
    plt.close()

if __name__ == '__main__':
    print '------------------------------------------------------------'
    sr = 22050
    alpha = 0.01
    hop_step = 0.010
    audioCount = 8732

    plotPath = '/home/pdl/Desktop/media/plot/UrbanSound8K-plot/plt{:s}'.format(str(alpha))
    audioPath = '/home/pdl/Desktop/media/datasets/UrbanSound8K/audio'
    audioListFile = 'files/meta.audiolist'
    vadPath = 'vad-dict'
    vadFile = 'UrbanSound8K-vad{:s}.dict'.format(str(alpha))

    make_path(vadPath)
    make_path(plotPath)
    check_path(audioPath)
    check_file(audioListFile)

    vadDict = {}

    with open(audioListFile, 'rb') as f:
        lines = f.readlines()
        f.close()

        for idx, items in enumerate(lines):
            items = items.split('\n')[0].split('\t')
            audioName, _ = os.path.splitext(os.path.basename(items[0]))
            audioClassID = int(items[1])

            pltName = classid2name[audioClassID] + ' | ' + audioName + '.png'
            pltSave = os.path.join(plotPath, pltName)
            audioFile = os.path.join(audioPath, items[0])

            if not os.path.isfile(audioFile):
                raise IOError('cannot found file: {:s}'.format(audioFile))
            y, sr = librosa.load(audioFile, sr=sr, mono=True)

            #mark = np.zeros(int(1.0 * len(y) / sr / hop_step) + 1)
            _, rmse = fbank(y, sr, 0.025, 0.01, 40, int(sr * 0.025))
            thread = np.mean(rmse) * alpha
            mark = rmse > thread
            vadDict[audioName] = mark
            plot_figure(y, sr, mark, pltName, pltSave)

            print idx+1, items, audioName, mark.shape
            #break

        assert audioCount == len(vadDict)
        pickle.dump(vadDict, open(os.path.join(vadPath, vadFile), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)
        print '\nvad saved in {:s}'.format(os.path.join(vadPath, vadFile))