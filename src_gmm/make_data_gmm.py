import sys
sys.path.append('../')
from util import *

def look_vad_dict(vadDict):
    for key in vadDict:
        if key != '1-30344-A':continue
        print key, type(vadDict[key]), vadDict[key].shape
        # 5-170338-A <type 'numpy.ndarray'> (510,)
        print vadDict[key]
        break

def process(files, mode):
    make_path(dataPath)
    check_path(feaPath)
    check_file(files)
    lines = open(files, 'rb').readlines()
    silentData = []
    voiceData = []
    for items in lines:
        items = items.split('\n')[0].split('\t')
        audioClass, audioName = os.path.split(items[0])
        audioName, _ = os.path.splitext(audioName)
        audioID = int(items[1])

        check_file(os.path.join(feaPath, audioName+'.fea'))
        data = pickle.load(open(os.path.join(feaPath, audioName+'.fea'), 'rb'))
        if audioName not in vadDict:
            raise IOError('[{:s}] not in vadDict'.format(audioName))

        vad = vadDict[audioName]
        for i in range(min(len(vad), data.shape[1])):
            if vad[i] == 1:
                voiceData.append(data[:, i])
            else:
                silentData.append(data[:, i])

        #print audioClass, audioName, audioID, data.shape
        #break
    dataDict = {}
    dataDict['silent'] = silentData
    dataDict['voice'] = voiceData
    pickle.dump(dataDict, open(os.path.join(dataPath, mode+'.dict'), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    print '------------------------------------------------------------'
    dims = 64
    fold = 'fold1'
    feaPath = '../../feature/feature_ESC-10/22050_logspec'
    dataPath = '../../data/data_GMM/{:s}'.format(fold)
    trainFile = 'files/evaluate-setup/{:s}_train.txt'.format(fold)
    validFile = 'files/evaluate-setup/{:s}_valid.txt'.format(fold)
    testFile = 'files/evaluate-setup/{:s}_test.txt'.format(fold)
    vadFile = 'files/vad2numpy.dict'

    vadDict = pickle.load(open(vadFile, 'rb'))
    #look_vad_dict(vadDict)

    process(trainFile, 'train')
    process(validFile, 'valid')
    process(testFile, 'test')
    print '[Done]'

