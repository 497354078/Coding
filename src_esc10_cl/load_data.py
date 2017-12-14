import sys
sys.path.append('../')
from util import *

class LoadTrainData(Dataset):
    def __init__(self, files):
        self.data, self.labs = self.load_data(files)

    def __len__(self):
        return len(self.labs)

    def __getitem__(self, item):
        return self.data[item], self.labs[item]

    def load_data(self, files):
        print '----------------------------------------------------------------'
        print 'Load Dataset from [{:s}]'.format(files)
        check_file(files)
        dataDict = pickle.load(open(files, 'rb'))

        data = []
        labs = []
        for audioID in dataDict:
            for subData in dataDict[audioID]:
                if subData is None:
                    raise IOError('spk: [{:s}] data is None'.format(audioID))
                index = 0
                frame = 65
                step = 1
                assert subData.shape[0] == 64
                while index + frame <= subData.shape[1]:
                    tmpData = subData[:, index:index+frame]
                    tmpData = tmpData.reshape((1, tmpData.shape[0], tmpData.shape[1]))
                    data.append(tmpData)
                    labs.append(audioID)
                    index += step

        print 'data: ', len(data), data[0].shape
        print 'labs: ', len(labs)
        print ''
        return data, labs


class LoadValidData(Dataset):
    def __init__(self, files):
        self.data, self.labs = self.load_data(files)

    def __len__(self):
        return len(self.labs)

    def __getitem__(self, item):
        return self.data[item], self.labs[item]

    def load_data(self, files):
        print '----------------------------------------------------------------'
        print 'Load Dataset from [{:s}]'.format(files)
        check_file(files)
        dataDict = pickle.load(open(files, 'rb'))

        data = []
        labs = []
        for audioID in dataDict:
            for subData in dataDict[audioID]:
                if subData is None:
                    raise IOError('spk: [{:s}] data is None'.format(audioID))
                index = 0
                frame = 65
                step = 1
                assert subData.shape[0] == 64
                dataList = []
                labsList = []
                while index + frame <= subData.shape[1]:
                    tmpData = subData[:, index:index+frame]
                    tmpData = tmpData.reshape((1, tmpData.shape[0], tmpData.shape[1]))
                    dataList.append(tmpData)
                    labsList.append(audioID)
                    index += step
                dataMat = np.asarray(dataList)
                labsMat = np.asarray(labsList)
                dataTensor = torch.from_numpy(dataMat).type(torch.FloatTensor)
                labsTensor = torch.from_numpy(labsMat).type(torch.LongTensor)
                data.append(dataTensor)
                labs.append(labsTensor)

        print 'data: ', len(data), data[0].size()
        print 'labs: ', len(labs)
        print ''
        return data, labs


if __name__ == '__main__':
    print '----------------------------------------------------------------'

    dims = 64
    fold = 'fold0'
    dataPath = '../data/data_ESC-10/{:s}'.format(fold)
    trainFile = os.path.join(dataPath, 'train.dict')
    validFile = os.path.join(dataPath, 'valid.dict')

    trainData = LoadTrainData(trainFile)
    trainLoad = torch.utils.data.DataLoader(trainData, batch_size=512, shuffle=True, num_workers=8)

    for data, labs in trainLoad:
        print 'data: ', type(data), data.size()
        print 'labs: ', type(labs), labs.size()
        break

    validData = LoadValidData(validFile)

    for id_, (data, labs) in enumerate(validData):
        print 'data: ', type(data), data.size()
        print 'labs: ', type(labs), labs.size()
        if id_>=1:break

