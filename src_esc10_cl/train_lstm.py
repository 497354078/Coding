import sys
sys.path.append('../')
from util import *
from model_lstm import LSTM
from load_data_lstm import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch BasicCNN For Far ESC-10')
parser.add_argument('--num_classes', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
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

def train_model(saveFile):
    net.train()
    stime = time.time()
    for id_, (data, labs) in enumerate(trainLoad):
        #data = data.resize_(data.size(0), 1, data.size(1), data.size(2))
        data, labs = Variable(data.cuda()), Variable(labs.cuda())

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        _, pred = net(data)
        loss = F.cross_entropy(pred, labs)
        loss.backward()
        optimizer.step()

        # ACC
        _, pred = torch.max(pred.data, 1)
        acc = (pred == labs.data).sum()

        if (id_ + 1) % args.log_interval == 0:
            printStr = "Train [%d|%d] lr: %f Loss: %.4f Acc: %.4f BatchTime: %f" % (
                epoch + 1, (id_ + 1)*args.batch_size, lr, loss.data[0],
                1.0*acc/labs.data.size(0), time.time()-stime)
            rePrint(printStr)

    torch.save(net, saveFile)


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
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    timeMark = str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    #timeMark = '2017-12-14 18:07:47'
    make_path('log')
    logName = os.path.basename(sys.argv[0])#+'-'+timeMark
    logging.basicConfig(level=logging.INFO,
                    filename='log/{:s}.log'.format(logName),
                    filemode='a',
                    format='%(asctime)s : %(message)s')

    rePrint('-----------------------------------------------------------------')
    rePrint(str(args))

    fold = 'fold3'
    dataPath = '../../data/data_ESC-10/{:s}'.format(fold)
    modelPath = '../../model/model_ESC-10/{:s}'.format(fold)
    modelFile = 'LSTM.{:d}.model'
    make_path(modelPath)

    trainData = LoadTrainData(os.path.join(dataPath, 'train.embed'))
    validData = LoadValidData(os.path.join(dataPath, 'valid.embed'))
    trainLoad = torch.utils.data.DataLoader(trainData,
                    batch_size=args.batch_size, shuffle=True, num_workers=8)

    net = LSTM(embedding_dim=128, hidden_dim=128, num_classes=args.num_classes)
    net.cuda()

    criterion = nn.CrossEntropyLoss()
    criterion.cuda()

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7,
                            patience=5, verbose=True, threshold=1e-6,
                        threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]['lr']
        train_model(os.path.join(modelPath, modelFile.format(epoch+1)))
        eval_loss = valid_model()
        scheduler.step(eval_loss)
        rePrint('-----------------------------------------------------------------')

    rePrint('[Done]')
    rePrint('-----------------------------------------------------------------')


