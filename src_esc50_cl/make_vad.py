import os
import sys
import numpy as np
import cPickle as pickle

vadDict = pickle.load(open('vad-dict/ESC-50-vad0.01.dict', 'rb'))
for audioName in vadDict:
    print audioName, type(vadDict[audioName]),vadDict[audioName].shape
    break

def process(mark):
    newMark = np.zeros(mark.shape)
    for i in range(len(mark)):
        if mark[i] == True:
            for j in range(5):
                newMark[max(0, i-j-1)] = 1
                newMark[min(i+j+1, len(mark)-1)] = 1
    return newMark

metaFile = 'files/meta.audiolist'
vadFile = 'files/vad2numpy.0.01.dict'
vad2numpy = {}
for audioName in vadDict:
    vad2numpy[audioName] = process(vadDict[audioName])
pickle.dump(vad2numpy, open(vadFile, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

