import os
import sys
import time
import numpy
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from torchvision import transforms
from load_data import *
from util import *


def eval_model(net):
    net.eval()
    eval_loss = 0
    eval_cnt = 0
    eval_sum = 0
    eval_acc = 0
    seg_sum = 0
    seg_acc = 0
    prb_sum = 0
    prb_acc = 0
    for idx, (data, labs) in enumerate(testData):
        #data = data.resize_(data.size(0), 1, data.size(1), data.size(2))
        stime = time.time()
        if torch.cuda.is_available():
            data = data.cuda()
            labs = labs.cuda()
        data, labs  = Variable(data), Variable(labs)

        # Forward
        net.eval()
        outputs = net(data)
        loss = F.cross_entropy(outputs, labs)

        if torch.cuda.is_available():
            lossVal = loss.data.cpu().numpy()
        else:
            lossVal = loss.data.numpy()
        eval_loss += lossVal
        eval_cnt += 1

        # Acc
        _, pred = torch.max(outputs.data, 1)
        acc = (pred == labs.data).sum()

        #print("EvalBatch [%d|%d] Acc: %.4f BatchTime: %f\t" % (
        #    epoch + 1, idx + 1, 1.0*acc/labs.data.size(0), time.time()-stime))
        #sys.stdout.flush()

        # Slip
        eval_sum += labs.data.size(0)
        eval_acc += acc

        # Seg Voting
        if torch.cuda.is_available():
            pred = pred.cpu().numpy()
            labs = labs.data.cpu().numpy()
        cnt = np.zeros(n_class)
        for item in pred:
            cnt[item] += 1
        if np.argmax(cnt) == labs[0]:
            seg_acc += 1
        seg_sum += 1

        # Mean Prob
        if torch.cuda.is_available():
            prob = outputs.data.cpu().numpy()
        prob = prob.mean(axis=0)
        assert len(prob) == n_class
        if np.argmax(prob) == labs[0]:
            prb_acc += 1
        prb_sum += 1

    #rePrint('-----------------------------------------------------------')
    rePrint('epoch: %d AvrEval: %f SegAvr: %f ProbAvr: %f eval_loss: %f' % (
        epoch, 1.0*eval_acc/eval_sum, 1.0*seg_acc/seg_sum,
        1.0*prb_acc/prb_sum, eval_loss/eval_cnt))
    #rePrint('-----------------------------------------------------------')
    return eval_loss/eval_cnt



if __name__ == '__main__':
    print '-----------------------------------------------------------'
    fold = 4
    alpha = 0.01
    logging.basicConfig(filename='logs/test_resnet.{:d}.log.vad{:s}'.format(fold, str(alpha)),
                        level=logging.INFO)
    rePrint('-----------------------------------------------------------')
    sr = 22050
    n_class = 50
    hop_step = 0.010
    audioCount = 2000

    dataPath = '/aifs1/users/lj/project/data/data_ESC-50/{:d}_{:d}'.format(sr, fold)
    modelPath = '/aifs1/users/lj/project/model/model_ESC-50/{:s}_{:d}'.format(str(sr), fold)
    modelName = 'resnet.{:d}.model.vad{:s}'

    check_path(dataPath)
    check_path(modelPath)

    testDataFile = '{:d}x{:x}x{:d}.{:s}.vad{:s}'.format(
        vectorDims, sampleSize, sampleStep, 'test', str(alpha))
    testData = validDataHelper(dataPath, testDataFile)

    for epoch in range(20, 100, 3):

        modelFile = os.path.join(modelPath, modelName.format(epoch, str(alpha)))
        net = torch.load(modelFile)
        eval_model(net)
