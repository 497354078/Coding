import os
import sys

if __name__ == '__main__':
    print '-------------------------------------------------------------'
    audioPath = '/home/pdl/Desktop/media/datasets/UrbanSound8K/audio'
    filePath = 'files'
    evaluatePath = 'files/evaluate-setup'
    metaFile = 'meta.audiolist'

    if not os.path.exists(filePath):
        os.makedirs(filePath)
    if not os.path.exists(evaluatePath):
        os.makedirs(evaluatePath)

    f = open(os.path.join(filePath, metaFile), 'wb')
    dirDict = {}
    for directory in sorted(os.listdir(audioPath)):
        absDirectory = os.path.join(audioPath, directory)
        if not (os.path.isdir(absDirectory) and os.path.basename(absDirectory)[-1].isdigit()):
            continue
        #print directory
        fileList = []
        for audioFile in sorted(os.listdir(absDirectory)):
            if not audioFile.endswith('.wav'):
                continue
            audioID = int(audioFile.split('-')[1])
            assert (0 <= audioID and audioID <= 9)
            print directory, audioFile, audioID
            f.write(directory+'/'+audioFile+'\t'+str(audioID)+'\n')
            fileList.append((directory, audioFile, audioID))
        dirDict[directory] = fileList
    f.close()

    for idx, fold in enumerate(((10, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10))):
        validFold = fold[0]
        testFold = fold[1]
        ftrain = open(os.path.join(evaluatePath, 'fold{:d}_train.txt').format(idx), 'wb')
        fvalid = open(os.path.join(evaluatePath, 'fold{:d}_valid.txt').format(idx), 'wb')
        ftest = open(os.path.join(evaluatePath, 'fold{:d}_test.txt').format(idx), 'wb')
        for dirKey in dirDict:
            for directory, audioFile, audioID in dirDict[dirKey]:
                fwstr = directory+'/'+audioFile+'\t'+str(audioID)+'\n'
                foldID = directory.split('fold')[1]
                assert (1<=int(foldID) and int(foldID)<=10)
                if foldID == str(validFold):
                    fvalid.write(fwstr)
                elif foldID == str(testFold):
                    ftest.write(fwstr)
                else:
                    ftrain.write(fwstr)
        ftrain.close()
        fvalid.close()
        ftest.close()
