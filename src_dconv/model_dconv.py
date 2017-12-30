import sys
sys.path.append('../')
from util import *
from load_random_data import LoadRandomData

class DeConv(nn.Module):
    def __init__(self, num_classes):
        super(DeConv, self).__init__()

        self.num_classes = num_classes
        self.N_conv = 5

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1, bias=True)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1, bias=True)
        self.conv4 = nn.Conv2d(256, 512, 3, 2, 1, bias=True)
        self.conv5 = nn.Conv2d(512, 1024, 3, 2, 1, bias=True)

        self.dconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, bias=True)
        self.dconv2 = nn.ConvTranspose2d(512, 256, 3, 2, 1, bias=True)
        self.dconv3 = nn.ConvTranspose2d(256, 128, 3, 2, 1, bias=True)
        self.dconv4 = nn.ConvTranspose2d(128, 64, 3, 2, 1, bias=True)
        self.dconv5 = nn.ConvTranspose2d(64, 1, 3, 2, 1, bias=True)

        self.FLAG = True
        self.SizeQueue = []

    def forward(self, x):
        if self.FLAG:
            print 'init x: ', x.size()
            self.SizeQueue.append(x.size())

        for i in range(self.N_conv):
            conv_name = 'self.conv{:d}'.format(i+1)
            x = eval(conv_name)(x)
            if self.FLAG:
                print 'conv{:d} x: '.format(i+1), x.size()
                self.SizeQueue.append(x.size())

        SizeCnt = len(self.SizeQueue)-2

        for i in range(self.N_conv):
            dconv_name = 'self.dconv{:d}'.format(i+1)
            x = eval(dconv_name)(x, output_size=self.SizeQueue[SizeCnt])
            SizeCnt -= 1
            if self.FLAG: print 'dconv{:d} x: '.format(i+1), x.size()

        self.FLAG = False
        return x

if __name__ == '__main__':

    print ''
    num_classes = 10
    trainData = LoadRandomData(N=100, num_classes=num_classes)
    trainLoad = torch.utils.data.DataLoader(trainData,  batch_size=20, shuffle=True, num_workers=4)

    net = DeConv(num_classes=num_classes)

    for _, (data, labs) in enumerate(trainLoad):
        data, labs = Variable(data), Variable(labs)
        print type(data), data.size()
        print type(labs), labs.size()

        out = net(data)

        break

