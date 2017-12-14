import os
import sys
import time
import numpy as np
import cPickle as pickle
from util import *


def get_marklist(mark):
    assert len(mark) >= 2
    pType = mark[0]
    sPoint = 0
    marklist = []
    for i in range(len(mark)):
        if i == len(mark)-1:
            marklist.append((sPoint, i, pType))
        if mark[i] != pType:
            marklist.append((sPoint, i-1, pType))
            sPoint = i
            pType = mark[i]
    #for sPoint, ePoint, pType in marklist:
    #    print sPoint, ePoint, pType,
    #print ''
    return marklist

def smooth(marklist):
    tmp = []
    for sPoint, ePoint, pType in marklist:
        if ePoint-sPoint+1 <= 4 and pType == False:# !!!
            pType = not pType
        if pType == True:
            tmp.append((sPoint, ePoint))
            # print sPoint, ePoint, pType,
    #print ''
    assert len(tmp) > 0
    res = []
    sPoint, ePoint = tmp[0]
    for i in range(1, len(tmp)):
        sp, ep = tmp[i]
        if ePoint+1 == sp:
            ePoint = ep
        else:
            if ePoint-sPoint+1 >= 4:
                res.append((sPoint, ePoint))
            sPoint, ePoint = tmp[i]

    if ePoint - sPoint + 1 >= 4:
        res.append((sPoint, ePoint))

    #for sp, ep in res:
    #    print sp, ep,
    #print ''

    return res

if __name__ == '__main__':
    print '------------------------------------------------------------'
    alpha = 0.01
    hop_step = 0.010
    audioCount = 8732

    vadDictFile = 'vad-dict/UrbanSound8K-vad{:s}.dict'.format(str(alpha))
    vadDictListFile = 'vad-dict/UrbanSound8K-vad{:s}.dictlist'.format(str(alpha))
    vadDictListSmoothFile = 'vad-dict/UrbanSound8K-vad{:s}.dictlistsmooth'.format(str(alpha))

    vadDict = {}
    vadDictList = {}
    vadDictListSmooth = {}

    check_file(vadDictFile)
    silentDict = pickle.load(open(vadDictFile, 'rb'))

    cnt = 0
    for idx, audioName in enumerate(silentDict):

        audioMarkList = get_marklist(silentDict[audioName])
        audioMarkListSmooth = smooth(audioMarkList)

        print idx+1, audioName, silentDict[audioName].shape, \
            len(audioMarkList), len(audioMarkListSmooth)

        assert len(audioMarkList) > 0
        assert len(audioMarkListSmooth) > 0

        vadDictList[audioName] = audioMarkList
        vadDictListSmooth[audioName] = audioMarkListSmooth

        cnt += 1
        #if idx >= 6:exit(0)
    assert cnt == audioCount

    pickle.dump(vadDictList, open(vadDictListFile, 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(vadDictListSmooth, open(vadDictListSmoothFile, 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    print 'saved in {:s}'.format(vadDictListFile)
    print 'saved in {:s}'.format(vadDictListSmoothFile)
