import os
import sys
import copy
import numpy as np
import cPickle as pickle
from util import *

# ====== slip parameter =========================================
FLIP = False
vectorDims = 40
sampleSize = 80
sampleStep = 1

def get_samplelist(audioName, audioData):

    sampleList = []
    for sPoint, ePoint in vadDictList[audioName]:
        #print sPoint, ePoint
        ePoint = min(ePoint, audioData.shape[1]-1)
        assert sPoint < ePoint
        assert (ePoint-sPoint+1) >= 5

        if ePoint-sPoint+1 <= sampleSize:
            tmp = np.zeros((vectorDims, sampleSize))
            j = sPoint
            for i in range(sampleSize):
                tmp[:, i] = audioData[:, j]
                j += 1
                if j == ePoint+1:
                    j = sPoint
            assert tmp.shape == (vectorDims, sampleSize)
            sampleList.append(tmp)
            if FLIP == True:
                sampleList.append(np.fliplr(tmp))

        else:
            while sPoint + sampleSize <= ePoint:
                tmp = audioData[:, sPoint:sPoint+sampleSize]
                assert tmp.shape == (vectorDims, sampleSize)
                sampleList.append(tmp)
                if FLIP == True:
                    sampleList.append(np.fliplr(tmp))
                sPoint += sampleStep
    #for sample in sampleList:
    #    print sample.shape
    assert len(sampleList) > 0
    return sampleList

def create_data(inputFile, mode):
    check_file(inputFile)

    dataDict = {}
    with open(inputFile, 'rb') as f:
        lines = f.readlines()
        f.close()
        cnt = 0
        for idx, items in enumerate(lines):
            items = items.split('\n')[0].split('\t')
            if len(items) < 2:
                continue
            audioName, _ = os.path.splitext(os.path.basename(items[0]))
            audioLab = int(items[1])

            ''' # len(sampleList) < 3
            if audioName == '3-142601-A' or audioName == '3-142605-A'\
                    or audioName == '4-156843-A' or audioName == '4-185415-A'\
                    or audioName == '3-170015-A' or audioName == '4-182395-A'\
                    or audioName == '3-163459-A':
               continue
            '''
            audioFile = os.path.join(feaPath, audioName+'.fea')
            check_file(audioFile)
            audioData = pickle.load(open(audioFile, 'rb')).T

            cnt += 1
            #print idx+1, cnt, items, audioName, audioLab, audioData.shape,
            assert audioData.shape[1] >= 363 and audioData.shape[1] <= 722

            sampleList = get_samplelist(audioName, audioData)
            dataDict[audioName] = (audioLab, sampleList)
            #print len(sampleList)
            assert len(sampleList) >= 1
            #exit(0)
        assert len(dataDict) == cnt

    outputFile = os.path.join(dataPath, '{:d}x{:x}x{:d}.{:s}.vad{:s}'.format(
        vectorDims, sampleSize, sampleStep, mode, str(alpha)))
    if FLIP == True:
        outputFile += '.fliplr'
    pickle.dump(dataDict, open(outputFile, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    print 'data saved in {:s}\n'.format(outputFile)



if __name__ == '__main__':
    print '------------------------------------------------------------'
    sr = 22050
    fold = 4
    alpha = 0.01
    #second = 5
    hop_step = 0.010
    audioCount = 2000
    #frame_size = int(second / hop_step)+1

    trainFile = 'files/evaluate-setup/fold{:d}_train.txt'.format(fold)
    validFile = 'files/evaluate-setup/fold{:d}_valid.txt'.format(fold)
    testFile = 'files/evaluate-setup/fold{:d}_test.txt'.format(fold)
    vadDictListFile = 'vad-dict/ESC-50-vad{:s}.dictlistsmooth'.format(str(alpha))

    feaPath = '/aifs1/users/lj/project/feature/feature_ESC-50/22050/logfbank'
    dataPath = '/aifs1/users/lj/project/data/data_ESC-50/{:d}_{:d}'.format(sr, fold)

    check_file(trainFile)
    check_file(validFile)
    check_file(testFile)
    check_file(vadDictListFile)

    check_path(feaPath)
    make_path(dataPath)

    vadDictList = pickle.load(open(vadDictListFile, 'rb'))
    create_data(trainFile, 'train')
    create_data(validFile, 'valid')
    create_data(testFile, 'test')

