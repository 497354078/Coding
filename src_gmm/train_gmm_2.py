import sys
sys.path.append('../')
from util import *

import sklearn
from sklearn.mixture import GaussianMixture

def train_gmm(files):
    check_file(files)
    trainDict = pickle.load(open(files, 'rb'))
    silentData = np.asarray(trainDict['silent'])
    voiceData = np.asarray(trainDict['voice'])

    estimator_silent = GaussianMixture(n_components=11,
                   covariance_type='diag', max_iter=100, random_state=0)
    #estimator_silent.means_init = silentData.mean(axis=0).reshape((1, dims))
    estimator_silent.fit(silentData)

    estimator_voice = GaussianMixture(n_components=11,
                   covariance_type='diag', max_iter=100, random_state=0)
    #estimator_voice.means_init = voiceData.mean(axis=0).reshape((1, dims))
    estimator_voice.fit(voiceData)

    estimator = {}
    estimator['silent'] = estimator_silent
    estimator['voice'] = estimator_voice
    pickle.dump(estimator, open(os.path.join(modelPath, 'gmm.model'), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)

def eval_gmm(estimator, files):
    check_file(files)
    evalDict = pickle.load(open(files, 'rb'))

    silentData = np.asarray(evalDict['silent'])
    pred_silent = estimator['silent'].score_samples(silentData)
    pred_voice = estimator['voice'].score_samples(silentData)
    print pred_silent.min(), pred_silent.max(), pred_silent
    print pred_voice.min(), pred_voice.max(), pred_voice
    judge = pred_silent > pred_voice
    print len(judge), sum(judge), len(judge)-sum(judge), 1.0*sum(judge)/len(judge)

    voiceData = np.asarray(evalDict['voice'])
    pred_silent = estimator['silent'].score_samples(voiceData)
    pred_voice = estimator['voice'].score_samples(voiceData)
    judge = pred_silent < pred_voice
    print len(judge), sum(judge), len(judge)-sum(judge), 1.0*sum(judge)/len(judge)


if __name__ == '__main__':
    print '------------------------------------------------------------'
    dims = 64
    fold = 'fold1'
    dataPath = '../../data/data_GMM/{:s}'.format(fold)
    waitPath = '../../data/data_wait'
    modelPath = '../../model/model_GMM/{:s}'.format(fold)
    make_path(modelPath)

    train_gmm(os.path.join(dataPath, 'train.dict'))

    estimator = pickle.load(open(os.path.join(modelPath, 'gmm.model')))

    eval_gmm(estimator, os.path.join(dataPath, 'valid.dict'))
    eval_gmm(estimator, os.path.join(dataPath, 'test.dict'))


