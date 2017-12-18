import sys
sys.path.append('../')
from util import *


class Basic_CNN(nn.Module):
    def __init__(self, num_classes):
        super(Basic_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(64, 6), stride=(1, 1), padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(1, 60), stride=(1, 1), padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256*1, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, num_classes)

        self.relu = nn.ReLU()
        self.drop1d = nn.Dropout(0.5)
        self.drop2d = nn.Dropout2d(0.5)

        self.flag = True

    def forward(self, x):
        if self.flag: print 'raw x: ', x.size()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop2d(x)
        #x = self.maxpool(x)
        if self.flag: print 'conv1 x: ', x.size()

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop2d(x)
        #x = self.maxpool(x)
        if self.flag: print 'conv2 x: ', x.size()

        x = x.view(x.size(0), -1)
        y = copy.copy(x)
        if self.flag: print 'view x: ', x.size()

        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop1d(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.drop1d(x)

        x = self.fc3(x)
        self.flag = False
        return y, x

