import sys
sys.path.append('../')
from util import *

import sklearn
from sklearn.mixture import GaussianMixture

def eval_gmm(estimator, data):

    #pred_silent = estimator['silent'].score_samples(data)
    #pred_voice = estimator['voice'].score_samples(data)
    #judge = pred_silent > pred_voice
    #print len(judge), sum(judge), len(judge)-sum(judge), 1.0*sum(judge)/len(judge)

    pred_silent = estimator['silent'].score_samples(data)
    pred_voice = estimator['voice'].score_samples(data)
    judge = pred_silent < pred_voice
    #print len(judge), sum(judge), len(judge)-sum(judge), 1.0*sum(judge)/len(judge)
    return judge

def process(mark):
    newMark = np.zeros(mark.shape)
    for i in range(len(mark)):
        if mark[i] == True:
            for j in range(5):
                newMark[max(0, i-j-1)] = 1
            for j in range(7):
                newMark[min(i+j+1, len(mark)-1)] = 1
    return newMark

if __name__ == '__main__':
    print '------------------------------------------------------------'
    sr = 22050
    dims = 64
    modelPath = '../../model/model_GMM/fold'
    check_path(modelPath)
    estimator = pickle.load(open(os.path.join(modelPath, 'gmm.model')))

    featPath = '../../feature/feature_ESC-50/{:d}_logspec'.format(sr)
    metaFile = '../src_esc50_cl/files/meta.audiolist'
    check_file(metaFile)

    vadFile = '../src_esc50_cl/files/vad2numpy.dict'

    vad2numpy = {}
    lines = open(metaFile, 'rb').readlines()
    for idx, items in enumerate(lines):
        items = items.split('\n')[0].split('\t')
        audioName, _ = os.path.splitext(os.path.basename(items[0]))
        audioLabs = int(items[1])
        data = pickle.load(open(os.path.join(featPath, audioName+'.fea'), 'rb'))
        mark = eval_gmm(estimator, data.T)
        mark = process(mark)
        vad2numpy[audioName] = mark
        if idx <2:
            print 'items: ', items, 'mark: ', mark.shape
            print mark

    print 'vad2numpy: ', len(vad2numpy)
    pickle.dump(vad2numpy, open(vadFile, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


