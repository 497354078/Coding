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

    stime = time.time()
    print 'Train silent GMM..'
    estimator_silent = GaussianMixture(n_components=64,
                   covariance_type='diag', max_iter=200, random_state=0)
    #estimator_silent.means_init = silentData.mean(axis=0).reshape((1, dims))
    estimator_silent.fit(silentData)
    print ' [Finished]', time.time()-stime
    
    stime = time.time()
    print 'Train voice GMM..'
    estimator_voice = GaussianMixture(n_components=64,
                   covariance_type='diag', max_iter=200, random_state=0)
    #estimator_voice.means_init = voiceData.mean(axis=0).reshape((1, dims))
    estimator_voice.fit(voiceData)
    print ' [Finished]', time.time()-stime

    estimator = {}
    estimator['silent'] = estimator_silent
    estimator['voice'] = estimator_voice
    pickle.dump(estimator, open(os.path.join(modelPath, 'gmm.model'), 'wb'),
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

    pred_silent = estimator['silent'].score_samples(silentData)
    pred_voice = estimator['voice'].score_samples(silentData)
    judge = pred_silent > pred_voice
    print len(judge), sum(judge), len(judge)-sum(judge), 1.0*sum(judge)/len(judge)

    pred_silent = estimator['silent'].score_samples(voiceData)
    pred_voice = estimator['voice'].score_samples(voiceData)
    judge = pred_silent < pred_voice
    print len(judge), sum(judge), len(judge)-sum(judge), 1.0*sum(judge)/len(judge)


if __name__ == '__main__':
    print '------------------------------------------------------------'
    dims = 64
    fold = 'fold'
    dataPath = '../../data/data_GMM/fold0'
    modelPath = '../../model/model_GMM/{:s}'.format(fold)
    make_path(modelPath)

    train_gmm(dataPath)

    estimator = pickle.load(open(os.path.join(modelPath, 'gmm.model')))

    eval_gmm(estimator, dataPath)


