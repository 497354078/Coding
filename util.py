import os
import sys
import time
import copy
import pydub
import numpy
import numpy as np
import logging
import argparse
import cPickle
import cPickle as pickle
import librosa
import librosa.display
import matplotlib.pyplot as plt

import sklearn
import sklearn.metrics

from sklearn import preprocessing

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import euclidean_distances

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, roc_auc_score

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

def rePrint(string):
    print string
    logging.info(string)

def check_file(files):
    if not os.path.isfile(files):
        raise IOError('cannot found file: {:s}'.format(files))

def check_path(paths):
    if not os.path.exists(paths):
        raise IOError('cannot found path: {:s}'.format(paths))

def make_path(paths):
    if not os.path.exists(paths):
        os.makedirs(paths)


def plot_wave(y, sr=22050, logspec=None, mono=True, audioName=None, pltSave=None, show=True):

    plt.figure()

    plt.subplot(2, 1, 1)
    plt.ylim(-1, 1)
    librosa.display.waveplot(y, sr=sr)
    plt.title(audioName)

    plt.subplot(2, 1, 2)
    plt.ylim(0, 1)
    if logspec is not None:
        D = logspec
    else:
        if mono:
            S = librosa.stft(y)
        else:
            S = librosa.stft(y[0])
        D = librosa.amplitude_to_db(S, ref=np.max)
    librosa.display.specshow(D, x_axis='time', y_axis='log')
    plt.title('Log-frequency power spectrogram')

    plt.tight_layout()
    if pltSave is not None:
        plt.savefig(pltSave)
    if show == True:
        plt.show()
    plt.close()

