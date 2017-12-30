import os
import sys
import shutil

def write_dict(dict_, files):
    f = open(files, 'wb')
    for id_, key in enumerate(dict_):
        if id_ == 0: print key, dict_[key]
        s = key
        for items in dict_[key]:
            s += ' '+items
        s += '\n'
        f.write(s)
    f.close()

def change_spk2utt(spkDict, uttDict, spk2utt):
    newspk2utt = {}
    for spkName in spk2utt:
        newSpkName = spkDict[spkName][0]
        newspk2utt[newSpkName] = []
        for uttName in spk2utt[spkName]:
            newUttName = uttDict[uttName][0]
            newspk2utt[newSpkName].append(newUttName)
    return newspk2utt

def change_utt2spk(spkDict, uttDict, utt2spk):
    newutt2spk = {}
    for uttName in utt2spk:
        newutt2spk[uttDict[uttName][0]] = spkDict[utt2spk[uttName]][0]
    return newutt2spk

def change_wavscp(uttDict, wavscp, destPath):
    newwavscp = {}
    for uttName in wavscp:
        newUttName = uttDict[uttName][0]
        newwavscp[newUttName] = [os.path.join(destPath, newUttName+'.ogg')]
        shutil.copy(wavscp[uttName][0], os.path.join(destPath, newUttName+'.ogg'))
    return newwavscp

def make_files(files, dataPath, destPath, dataType):

    lines = open(files, 'rb').readlines()
    spk2utt = {}
    utt2spk = {}
    wavscp = {}
    for id_, items in enumerate(lines):
        items = items.split('\n')[0].split('\t')
        spkName, uttName = os.path.split(items[0])
        uttName = os.path.splitext(uttName)[0]
        #spkName = spkName.replace(' ', '_').replace('-', '_')
        #uttName = uttName.replace('-', '_')
        if id_ == 0: print items, spkName, uttName
        assert uttName not in wavscp
        wavscp[uttName] = [os.path.join(dataPath, items[0])]

        if spkName not in spk2utt:
            spk2utt[spkName] = []
        spk2utt[spkName].append(uttName)

        assert uttName not in utt2spk
        utt2spk[uttName] = [spkName]

    spkDict = {}
    uttDict = {}
    for spkName in spk2utt:
        newSpkName = 'ESC10_'+spkName.split()[0]
        assert spkName not in spkDict
        spkDict[spkName] = [newSpkName]
        for id_, uttName in enumerate(spk2utt[spkName]):
            newUttName = newSpkName+'U{:04d}'.format(id_)
            assert uttName not in uttDict
            uttDict[uttName] = [newUttName]

    spk2utt = change_spk2utt(spkDict, uttDict, spk2utt)
    utt2spk = change_utt2spk(spkDict, uttDict, utt2spk)
    wavscp = change_wavscp(uttDict, wavscp, destPath)

    if not os.path.exists(dataType):
        os.makedirs(dataType)
    write_dict(spk2utt, os.path.join(dataType, 'spk2utt'))
    write_dict(utt2spk, os.path.join(dataType, 'utt2spk'))
    write_dict(wavscp, os.path.join(dataType, 'wav.scp'))
    write_dict(spkDict, os.path.join(dataType, 'spkDict'))
    write_dict(uttDict, os.path.join(dataType, 'uttDict'))


if __name__ == '__main__':
    dataType = 'data_ESC-10'
    dataPath = '/home/lj/work/project/dataset/data_ESC-10'
    destPath = '/home/lj/work/project/dataset/data_audio'
    metaFile = '../src_esc10/files/meta.audiolist'

    make_files(metaFile, dataPath, destPath, dataType)


