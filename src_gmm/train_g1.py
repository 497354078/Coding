import sys
sys.path.append('../')
from util import *

import sklearn
from sklearn.mixture import GaussianMixture


def train_gmm(path):
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

    estimator = GaussianMixture(n_components=2,
                   covariance_type='diag', max_iter=100, random_state=11)
    meanInit = np.zeros((2, dims))
    meanInit[0] = silentData.mean(axis=0)
    meanInit[1] = voiceData.mean(axis=0)
    estimator.means_init = meanInit
    estimator.fit(np.concatenate((silentData, voiceData), axis=0))

    pickle.dump(estimator, open(os.path.join(modelPath, 'gmm1.model'), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)

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
    print voiceData
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
    dataPath = '../../data/data_GMM/fold0'
    modelPath = '../../model/model_GMM/{:s}'.format(fold)
    make_path(modelPath)

    train_gmm(dataPath)

    estimator = pickle.load(open(os.path.join(modelPath, 'gmm1.model')))
    print estimator.means_

    eval_gmm(estimator, dataPath)


