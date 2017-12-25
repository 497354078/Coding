import sys
sys.path.append('../')
from util import *

import sklearn
from sklearn.mixture import GaussianMixture

def process(mark):
    newMark = np.zeros(mark.shape)
    for i in range(len(mark)):
        if mark[i] == True:
            for j in range(5):
                newMark[max(0, i-j-1)] = 1
                newMark[min(i+j+1, len(mark)-1)] = 1
    return newMark

def train_gmm(estimator, feaPath, fileList):
    check_path(feaPath)
    data = []
    for files in fileList:
        check_file(files)
        lines = open(files, 'rb').readlines()
        for items in lines:
            items = items.split('\n')[0].split('\t')
            audioClass, audioName = os.path.split(items[0])
            audioName, _ = os.path.splitext(audioName)
            audioID = int(items[1])

            check_file(os.path.join(feaPath, audioName+'.fea'))
            tmpdata = pickle.load(open(os.path.join(feaPath, audioName+'.fea'), 'rb'))
            assert tmpdata.shape[0] == dims
            for i in range(tmpdata.shape[1]):
                data.append(tmpdata[:, i])
    data = np.asarray(data)

    gmm = GaussianMixture(n_components=2,
                   covariance_type='diag', max_iter=100, random_state=0)
    gmm.means_init = estimator.means_
    gmm.fit(data)

    pickle.dump(gmm, open(os.path.join(modelPath, 'gmm_esc50.model'), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    return gmm

def eval_gmm(estimator, path):
    check_path(path)
    trainDict = pickle.load(open(os.path.join(path, 'train.dict'), 'rb'))
    validDict = pickle.load(open(os.path.join(path, 'valid.dict'), 'rb'))
    testDict = pickle.load(open(os.path.join(path, 'test.dict'), 'rb'))

    silentData = np.concatenate((np.asarray(trainDict['silent']),
                                 np.asarray(validDict['silent'])), axis=0)
    silentData = np.concatenate((silentData, np.asarray(testDict['silent'])), axis=0)

    voiceData = np.concatenate((np.asarray(trainDict['voice']),
                                 np.asarray(validDict['voice'])), axis=0)
    voiceData = np.concatenate((voiceData, np.asarray(testDict['voice'])), axis=0)

    acc = 0
    cnt = 0

    silentData = np.asarray(silentData)
    pred = estimator.predict(silentData)
    acc += silentData.shape[0]-sum(pred)
    cnt += silentData.shape[0]

    voiceData = np.asarray(voiceData)
    pred = estimator.predict(voiceData)
    acc += sum(pred)
    cnt += voiceData.shape[0]

    print 'eval gmm: {:f}'.format(1.0*acc/cnt)

if __name__ == '__main__':
    print '------------------------------------------------------------'
    dims = 64
    fold = 'fold'
    modelPath = '../../model/model_GMM/{:s}'.format(fold)
    make_path(modelPath)
    estimator = pickle.load(open(os.path.join(modelPath, 'gmm1.model')))

    fold = 'fold0'
    feaPath = '../../feature/feature_ESC-50/22050_logspec'
    #fileList = ['files/evaluate-setup/{:s}_train.txt'.format(fold),
    #            'files/evaluate-setup/{:s}_valid.txt'.format(fold),
    #            'files/evaluate-setup/{:s}_test.txt'.format(fold) ]
    fileList = ['files/meta.audiolist']
    gmm = train_gmm(estimator, feaPath, fileList)
    eval_gmm(gmm, '../../data/data_GMM/fold0')
    eval_gmm(estimator, '../../data/data_GMM/fold0')

    metaFile = 'files/meta.audiolist'
    vadFile = 'files/vad2numpy.esc50.dict'
    vad2numpy = {}
    lines = open(metaFile, 'rb').readlines()
    for idx, items in enumerate(lines):
        items = items.split('\n')[0].split('\t')
        audioName, _ = os.path.splitext(os.path.basename(items[0]))
        audioLabs = int(items[1])
        data = pickle.load(open(os.path.join(feaPath, audioName+'.fea'), 'rb'))
        mark = gmm.predict(data.T)
        mark = process(mark)
        vad2numpy[audioName] = mark
        if idx <2:
            print 'items: ', items, 'mark: ', mark.shape
            print mark
    pickle.dump(vad2numpy, open(vadFile, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    print gmm.means_
