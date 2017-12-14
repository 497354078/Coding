import os
import sys
import numpy as np
import cPickle as pickle

if __name__ == '__main__':
    print '------------------------------------------------------------'
    second = 5
    hop_step = 0.010
    audioCount = 400
    frame_size = int(second / hop_step)+10
    silentLines = 906

    silentFile = 'files/ESC-10-silent'
    silentDictFile = 'files/ESC-10-silent.dict'
    silentDictListFile = 'files/ESC-10-silent.dictlist'

    silentDict = {}
    silentDictList = {}

    if not os.path.isfile(silentFile):
        raise IOError('cannot found file: {:s}'.format(silentFile))

    with open(silentFile, 'rb') as f:
        lines = f.readlines()
        f.close()

        sLine = 0
        for idx, items in enumerate(lines):
            items = items.split()
            if len(items) != 3:
                continue
            audioName = items[0]
            sPoint = int(float(items[1])/hop_step)
            ePoint = int(float(items[2])/hop_step)
            print audioName, sPoint, ePoint
            if audioName not in silentDict:
                silentDict[audioName] = np.zeros(frame_size)
                silentDictList[audioName] = []
            silentDictList[audioName].append((sPoint, ePoint))
            while sPoint <= ePoint:
                silentDict[audioName][sPoint] = 1
                sPoint += 1
            sLine += 1
            #exit(0)
        assert sLine == silentLines
        assert len(silentDict) == audioCount

    pickle.dump(silentDict, open(silentDictFile, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(silentDictList, open(silentDictListFile, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    print '\nsilentDict saved in {:s}'.format(silentDictFile)
    print '\nsilentDictList saved in {:s}'.format(silentDictListFile)