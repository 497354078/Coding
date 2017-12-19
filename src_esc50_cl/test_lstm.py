import sys
sys.path.append('../')
from util import *
from model_lstm import *
from load_data_lstm import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch BasicCNN For Far ESC-10')
parser.add_argument('--num_classes', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='M',
                    help='SGD momentum (default: 5e-4)')
parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 512)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()


def valid_model():
    net.eval()

    eval_loss = 0
    eval_cnt = 0
    eval_acc = 0
    eval_sum = 0
    seg_acc = 0
    seg_sum = 0
    prb_acc = 0
    prb_sum = 0
    stime = time.time()

    for idx, (data, labs) in enumerate(validData):
        #data = data.resize_(data.size(0), 1, data.size(1), data.size(2))
        data, labs = Variable(data.cuda()), Variable(labs.cuda())

        # Forward
        _, out = net(data)
        #loss = F.cross_entropy(pred, labs)
        loss = criterion(out, labs)
        loss = 1.0*loss.data.cpu().numpy()
        eval_loss += loss
        eval_cnt += 1

        # Acc Slip
        _, pred = torch.max(out.data, 1)
        acc = (pred == labs.data).sum()

        eval_acc += acc
        eval_sum += labs.data.size(0)

        # Seg Voting
        pred = pred.cpu().numpy()
        labs = labs.data.cpu().numpy()
        cnt = np.zeros(args.num_classes)
        for item in pred:
            cnt[item] += 1
        if np.argmax(cnt) == labs[0]:
            seg_acc += 1
        seg_sum += 1

        # Mean Prob
        prob = out.data.cpu().numpy()
        prob = prob.mean(axis=0)
        assert len(prob) == args.num_classes
        if np.argmax(prob) == labs[0]:
            prb_acc += 1
        prb_sum += 1


    rePrint('AvrEval: %f  SegAvr: %f ProbAvr: %f eval_loss: %f time: %f' % (
        1.0*eval_acc/eval_sum, 1.0*seg_acc/seg_sum,
        1.0*prb_acc/prb_sum, 1.0*eval_loss/eval_cnt,
        time.time()-stime))
    return eval_loss/eval_cnt



if __name__ == '__main__':
    print '-----------------------------------------------------------------'
    timeMark = str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    #timeMark = '2017-12-14 18:07:47'
    make_path('log')
    logName = os.path.basename(sys.argv[0])+'-'+timeMark
    logging.basicConfig(level=logging.INFO,
                    filename='log/{:s}.log'.format(logName),
                    filemode='a',
                    format='%(asctime)s : %(message)s')

    rePrint('-----------------------------------------------------------------')

    fold = 'fold4'
    dataPath = '../../data/data_ESC-10/{:s}'.format(fold)
    modelPath = '../../model/model_ESC-10/{:s}'.format(fold)
    modelFile = 'LSTM.{:d}.model'
    make_path(modelPath)

    validData = LoadValidData(os.path.join(dataPath, 'test.embed'))


    criterion = nn.CrossEntropyLoss()
    criterion.cuda()

    for epoch in range(args.epochs):
        net = torch.load(os.path.join(modelPath, modelFile.format(epoch+1)))
        net.cuda()
        eval_loss = valid_model()

    rePrint('[Done]')
    rePrint('-----------------------------------------------------------------')


