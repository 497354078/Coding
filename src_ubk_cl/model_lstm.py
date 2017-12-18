import sys
sys.path.append('../')
from util import *

class LSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, num_classes):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=1,
                            bias=True,
                            batch_first=True,
                            dropout=0.5,
                            bidirectional=True
                            )

        self.hidden2tag = nn.Linear(hidden_dim*2, num_classes)
        self.hidden = self.init_hidden()
        self.flag = True

    def init_hidden(self): # layers, batch_size, hidden
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, x):
        if self.flag:print 'input x:', x.size()
        x, _ = self.lstm(x)
        if self.flag:print 'lstm x:', x.size()
        x = torch.max(x, dim=1)[0]
        if self.flag:print 'tmax x:', x.size()
        x = x.view(x.size(0), -1)
        y = copy.copy(x)
        x = self.hidden2tag(x)
        self.flag = False
        return y, x

