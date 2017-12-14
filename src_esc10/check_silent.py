import os
import sys
import copy
import numpy as np
import cPickle as pickle
import librosa
import librosa.display
import matplotlib.pyplot as plt


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
    plt.title("hand-label")

    plt.tight_layout()
    plt.savefig(pltSave)
    #plt.show()
    plt.close()

if __name__ == '__main__':
    print '------------------------------------------------------------'
    sr = 22050
    second = 5
    hop_step = 0.010
    audioCount = 400
    frame_size = int(second / hop_step)+1
    silentLines = 906

    plotPath = '/home/pdl/Desktop/media/plot/ESC-10-silent'
    audioPath = '/home/pdl/Desktop/media/datasets/ESC-10/ESC-10-master'
    audioListFile = 'files/meta.audiolist'
    silentDictFile = 'files/ESC-10-silent.dict'

    if not os.path.isfile(silentDictFile):
        raise IOError('cannot found file: {:s}'.format(silentDictFile))
    if not os.path.isfile(audioListFile):
        raise IOError('cannot found file: {:s}'.format(audioListFile))
    if not os.path.exists(audioPath):
        raise IOError('cannot found path: {:s}'.format(audioPath))
    if not os.path.exists(plotPath):
        os.makedirs(plotPath)

    silentDict = pickle.load(open(silentDictFile, 'rb'))

    with open(audioListFile, 'rb') as f:
        lines = f.readlines()
        f.close()

        for idx, items in enumerate(lines):
            items = items.split('\n')[0].split('\t')
            audioName, _ = os.path.splitext(os.path.basename(items[0]))
            print items, audioName
            if audioName not in silentDict:
                raise IOError('silentDict cannot found audioName: {:s}'.format(audioName))

            audioFile = os.path.join(audioPath, items[0])
            if not os.path.isfile(audioFile):
                raise IOError('cannot found file: {:s}'.format(audioFile))
            y, sr = librosa.load(audioFile, sr=sr, mono=True)

            mark = silentDict[audioName][0:int(len(y) / sr / hop_step) + 1]
            pltSave = os.path.join(plotPath, audioName+'.png')

            plot_figure(copy.copy(y), sr, mark, audioName, pltSave)
            #exit(0)
