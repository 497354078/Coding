import sys
sys.path.append('../')
from util import *

import sklearn
from sklearn.mixture import GaussianMixture

def train_gmm(files):
    print 'Load train data & train gmm model'
    stime = time.time()
    check_file(files)
    trainDict = pickle.load(open(files, 'rb'))
    silentData = np.asarray(trainDict['silent'])
    voiceData = np.asarray(trainDict['voice'])
    trainData = np.concatenate((silentData, voiceData), axis=0)

    meanInit = np.zeros((2, dims))
    meanInit[0] = silentData.mean(axis=0)
    meanInit[1] = voiceData.mean(axis=0)

    estimator = GaussianMixture(n_components=2,
                   covariance_type='diag', max_iter=100, random_state=0)
    estimator.means_init = meanInit
    estimator.fit(trainData)

    pickle.dump(trainData.mean(), open(os.path.join(waitPath, 'train.mean'), 'wb'), 
        protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(trainData.std(), open(os.path.join(waitPath, 'train.std'), 'wb'),
        protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(estimator, open(os.path.join(modelPath, 'gmm.model'), 'wb'), 
        protocol=pickle.HIGHEST_PROTOCOL)
    print 'Finished train & saved gmm model in:\n   {:s}\nUsetime {:f}\n'.format(
            os.path.join(modelPath, 'gmm.model'), time.time()-stime)
    
def eval_gmm(estimator, files):
    print 'Load valid data & valid gmm model'
    check_file(files)
    evalDict = pickle.load(open(files, 'rb'))
    acc = 0
    cnt = 0
    silentData = np.asarray(evalDict['silent'])
    pred = estimator.predict(silentData)
    print type(pred), pred.shape, silentData.shape[0]-sum(pred), sum(pred)
    acc += silentData.shape[0]-sum(pred)
    cnt += silentData.shape[0]
    voiceData = np.asarray(evalDict['voice'])
    pred = estimator.predict(voiceData)
    print type(pred), pred.shape, sum(pred), voiceData.shape[0]-sum(pred)
    acc += sum(pred)
    cnt += voiceData.shape[0]
    print 'eval gmm: {:f}\n'.format(1.0*acc/cnt)


if __name__ == '__main__':
    print '------------------------------------------------------------'
    dims = 64
    fold = 'fold1'
    dataPath = '../../data/data_GMM/{:s}'.format(fold)
    modelPath = '../../model/model_GMM/{:s}'.format(fold)
    make_path(modelPath)

    train_gmm(os.path.join(dataPath, 'train.dict'))

    estimator = pickle.load(open(os.path.join(modelPath, 'gmm.model')))

    eval_gmm(estimator, os.path.join(dataPath, 'valid.dict'))
    eval_gmm(estimator, os.path.join(dataPath, 'test.dict'))


