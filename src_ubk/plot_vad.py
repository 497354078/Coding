import os
import sys
import copy
import numpy as np
import cPickle as pickle
import librosa
import librosa.display
import matplotlib.pyplot as plt
from util import *


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
    #second = 5
    hop_step = 0.010
    audioCount = 8732
    #frame_size = int(second / hop_step)+1

    plotPath = '/home/pdl/Desktop/media/plot/UrbanSound8K-plot/plt{:s}-smooth2'.format(str(alpha))
    audioPath = '/home/pdl/Desktop/media/datasets/UrbanSound8K/audio'
    audioListFile = 'files/meta.audiolist'
    vadDictFile = 'vad-dict/UrbanSound8K-vad{:s}.dictlistsmooth'.format(str(alpha))

    check_file(vadDictFile)
    check_file(audioListFile)
    check_path(audioPath)
    make_path(plotPath)

    silentDict = pickle.load(open(vadDictFile, 'rb'))

    with open(audioListFile, 'rb') as f:
        lines = f.readlines()
        f.close()

        for idx, items in enumerate(lines):
            items = items.split('\n')[0].split('\t')
            audioName, _ = os.path.splitext(os.path.basename(items[0]))
            audioClassID = int(items[1])
            #if audioName != '1-17565-A':
            #    continue

            pltName = classid2name[audioClassID] + ' | ' + audioName + '.png'
            pltSave = os.path.join(plotPath, pltName)
            #if os.path.isfile(pltSave):
            #    continue

            if audioName not in silentDict:
                raise IOError('silentDict cannot found audioName: {:s}'.format(audioName))

            audioFile = os.path.join(audioPath, items[0])
            if not os.path.isfile(audioFile):
                raise IOError('cannot found file: {:s}'.format(audioFile))
            y, sr = librosa.load(audioFile, sr=sr, mono=True)
            print idx+1, items, audioName, int(1.0*len(y) / sr / hop_step) + 1, ':',
            #mark = silentDict[audioName][0:int(len(y) / sr / hop_step) + 1]
            mark = np.zeros(int(1.0*len(y) / sr / hop_step) + 1)
            for sp, ep in silentDict[audioName]:
                print sp, ep,
                while sp <= ep:
                    mark[sp] = True
                    sp += 1
            print ''

            plot_figure(copy.copy(y), sr, mark, pltName, pltSave)
            #exit(0)
