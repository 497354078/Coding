import torch
import os
import sys
import time
import cPickle as pickle
import numpy as np
from torch.utils import data
from util import *

# ====== slip parameter =========================================
vectorDims = 40
sampleSize = 41
sampleStep = 1

class trainDataHelper(data.Dataset):
    def __init__(self, loadPath, loadFiles):
        self.images, self.labels = self.load_data(loadPath, loadFiles)

    def __getitem__(self, index):
        img, lab = self.images[index], self.labels[index]
        return img, lab

    def __len__(self):
        return len(self.labels)

    def load_data(self, loadPath, loadFiles):
        check_path(loadPath)
        print 'load {:s}'.format(os.path.join(loadPath, loadFiles))

        imgs, labs = [], []
        stime = time.time()

        files = os.path.join(loadPath, loadFiles)
        if not os.path.join(files):
            raise IOError('cannot found file: {:s}'.format(files))
        dataDict = pickle.load(open(files, 'rb'))
        for idx, key in enumerate(dataDict):
            classID, tmpData = dataDict[key]
            if len(tmpData) == 0:
                raise IOError('{:s} {:d}'.format(key, classID))

            for img in tmpData:
                imgs.append(img.reshape(1, img.shape[0], img.shape[1]).astype(np.float32))
                labs.append(int(classID))

        print ('\nLoad finished, usetime: {:f} {:d}'.format(time.time()-stime, len(labs)))
        print 'imgs: ', len(imgs), imgs[0].shape
        print 'labs: ', len(labs)
        return imgs, labs

class validDataHelper(data.Dataset):
    def __init__(self, loadPath, loadFiles):
        self.images, self.labels = self.load_data(loadPath, loadFiles)

    def __getitem__(self, index):
        img, lab = self.images[index], self.labels[index]
        return img, lab

    def __len__(self):
        return len(self.labels)

    def load_data(self, loadPath, loadFiles):
        check_path(loadPath)
        print 'load {:s}'.format(os.path.join(loadPath, loadFiles))

        imgs, labs = [], []
        stime = time.time()

        files = os.path.join(loadPath, loadFiles)
        if not os.path.join(files):
            raise IOError('cannot found file: {:s}'.format(files))
        dataDict = pickle.load(open(files, 'rb'))
        for key in dataDict:
            classID, tmpData = dataDict[key]
            if len(tmpData) == 0:
                raise IOError('{:s} {:d}'.format(key, classID))

            tmp1 = []
            tmp2 = []
            for img in tmpData:
                tmp1.append(img.reshape(1, img.shape[0], img.shape[1]).astype(np.float32))
                tmp2.append(classID)

            tmp1 = np.asarray(tmp1)
            tmp1 = torch.from_numpy(tmp1).type(torch.FloatTensor)

            tmp2 = np.asarray(tmp2)
            tmp2 = torch.from_numpy(tmp2).type(torch.LongTensor)

            imgs.append(tmp1)
            labs.append(tmp2)
        print ('\nLoad finished, usetime: {:f} {:d}'.format(time.time()-stime, len(labs)))
        print 'imgs: ', len(imgs), imgs[0].size()
        print 'labs: ', len(labs)
        return imgs, labs

if __name__ == '__main__':
    print '-----------------------------------------------------------'
    sr = 22050
    fold = 0
    alpha = 0.01
    hop_step = 0.010
    audioCount = 8732

    dataPath = '/aifs1/users/lj/project/data/data_UrbanSound8K/{:d}_{:d}'.format(sr, fold)

    trainDataFile = '{:d}x{:x}x{:d}.{:s}.vad{:s}'.format(
        vectorDims, sampleSize, sampleStep, 'train', str(alpha))
    trainData = trainDataHelper(dataPath, trainDataFile)
    trainLoad = torch.utils.data.DataLoader(trainData,  batch_size=100, shuffle=True, num_workers=4)

    for _, (imgs, labs) in enumerate(trainLoad):
        print (type(imgs), imgs.size())
        print (type(labs), labs.size())
        break

    validDataFile = '{:d}x{:x}x{:d}.{:s}.vad{:s}'.format(
        vectorDims, sampleSize, sampleStep, 'valid', str(alpha))
    validData = validDataHelper(dataPath, validDataFile)

    for _, (imgs, labs) in enumerate(validData):
        print (type(imgs), imgs.size())
        print (type(labs), labs.size())
        break


