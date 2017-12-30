import sys
sys.path.append('../')
from util import *

class LoadRandomData(Dataset):
    def __init__(self, N=10000, num_classes=10):
        self.N = N
        self.num_classes = num_classes
        self.data, self.labs = self.generate_random_data()

    def __getitem__(self, item):
        return self.data[item], self.labs[item]

    def __len__(self):
        return len(self.labs)

    def generate_random_data(self):
        data = []
        labs = []
        for i in range(self.N):
            data.append(np.random.rand(1, 64, 128).astype(np.float32))
            labs.append(i%self.num_classes)
        return data, labs


if __name__ == '__main__':
    print ''
    trainData = LoadRandomData(N=100, num_classes=10)
    trainLoad = torch.utils.data.DataLoader(trainData,  batch_size=20, shuffle=True, num_workers=4)

    for _, (imgs, labs) in enumerate(trainLoad):
        print type(imgs), imgs.size()
        print type(labs), labs.size()
        break

