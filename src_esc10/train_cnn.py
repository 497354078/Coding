import os
import sys
import time
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from torchvision import transforms
from load_data import *
from model_block import resnet, BasicCNN

import logging

def rePrint(printStr):
    print printStr
    logging.info(printStr)

def train_model(net, saveModel):
    net.train()
    train_acc = 0
    train_sum = 0
    train_loss = 0
    train_cnt = 0
    stime = time.time()
    for idx, (data, labs) in enumerate(trainLoad):
        #data = data.resize_(data.size(0), 1, data.size(1), data.size(2))

        if torch.cuda.is_available():
            data = data.cuda()
            labs = labs.cuda()
        data, labs  = Variable(data), Variable(labs)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = net(data)
        loss = F.cross_entropy(outputs, labs)
        loss.backward()
        optimizer.step()

        if torch.cuda.is_available():
            lossVal = loss.data.cpu().numpy()
        else:
            lossVal = loss.data.numpy()
        train_loss += lossVal
        train_cnt += 1

        # Acc
        _, pred = torch.max(outputs.data, 1)
        acc = (pred == labs.data).sum()
        #if (idx + 1) % steps == 0:
        #    rePrint("Train lr:%f [%d|%d] Loss: %.4f Acc: %.4f BatchTime: %f" % (
        #        lr, epoch + 1, idx + 1, loss.data[0],
        #           1.0*acc/labs.data.size(0), time.time()-stime))
        train_sum += labs.data.size(0)
        train_acc += acc
    rePrint("Train lr:%f [%d] Loss: %.4f Acc: %.4f EpochTime: %f" % (
        lr, epoch + 1, train_loss/train_cnt,
           1.0*train_acc/train_sum, time.time()-stime))

    #torch.save(net.state_dict(), saveModel)
    torch.save(net, saveModel)

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
    for idx, (data, labs) in enumerate(validData):
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
    rePrint('AvrEval: %f  SegAvr: %f ProbAvr: %f eval_loss: %f' % (
        1.0*eval_acc/eval_sum, 1.0*seg_acc/seg_sum,
        1.0*prb_acc/prb_sum, eval_loss/eval_cnt))
    #rePrint('-----------------------------------------------------------')
    return eval_loss/eval_cnt


if __name__ == '__main__':

    fold = 0
    logging.basicConfig(filename='logs/train_BasicCNN.{:d}.log.fliplr'.format(fold), level=logging.INFO)
    rePrint('-----------------------------------------------------------')
    n_class = 10
    sr = 22050
    second = 5
    hop_step = 0.010
    audioCount = 400
    frame_size = int(second / hop_step)+1
    silentLines = 906

    dataPath = '/home/pdl/Desktop/media/data/data_ESC-10/{:d}_{:d}'.format(sr, fold)
    trainDataFile = '{:d}x{:x}x{:d}.{:s}.fliplr'.format(
        vectorDims, sampleSize, sampleStep, 'train')

    trainData = trainDataHelper(dataPath, trainDataFile)
    trainLoad = torch.utils.data.DataLoader(trainData,  batch_size=256,
                                            shuffle=True, num_workers=4)

    validDataFile = '{:d}x{:x}x{:d}.{:s}.fliplr'.format(
        vectorDims, sampleSize, sampleStep, 'valid')
    validData = validDataHelper(dataPath, validDataFile)

    modelPath = '/home/pdl/Desktop/media/model/model_ESC-10/{:s}_{:d}'.format(str(sr), fold)

    rePrint('-----------------------------------------------------------')
    epochs = 100
    steps = 1
    #model = resnet(num_classes=n_class)
    model = BasicCNN(num_classes=n_class)
    if torch.cuda.is_available():
        model.cuda()
    lr = 0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min',
                                  factor=0.5, patience=5,
                                  verbose=False, threshold=1e-6)

    if not os.path.exists(modelPath):
        os.makedirs(modelPath)

    for epoch in range(epochs):
        lr = optimizer.param_groups[0]['lr']
        modelFile = os.path.join(modelPath, 'BasicCNN.{:d}.model.fliplr'.format(epoch))
        train_model(model, modelFile)
        eval_loss = eval_model(model)
        #print 'eval_loss: ', type(eval_loss), eval_loss
        scheduler.step(eval_loss)
        rePrint('-----------------------------------------------------------')
    rePrint('[Done]')
    rePrint('-----------------------------------------------------------')


