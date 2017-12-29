import os
import sys

def write_list(list_, files):
    f = open(files, 'wb')
    for id_, items in enumerate(list_):
        if id_ == 0: print items
        f.write(items)
    f.close()

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

def make_files(files, dataPath, dataType):

    lines = open(files, 'rb').readlines()
    wavscp = []
    spk2utt = {}
    utt2spk = {}
    for id_, items in enumerate(lines):
        items = items.split('\n')[0].split('\t')
        if id_ == 0: print items
        spkName, uttName = os.path.split(items[0])
        spkName = spkName.replace(' ', '_').replace('-', '_')
        uttName = os.path.splitext(uttName)[0].replace('-', '_')

        wavscp.append(uttName+' '+os.path.join(dataPath, items[0])+'\n')

        if spkName not in spk2utt:
            spk2utt[spkName] = []
        spk2utt[spkName].append(uttName)

        if uttName not in utt2spk:
            utt2spk[uttName] = []
        utt2spk[uttName].append(spkName)
    if not os.path.exists(dataType):
        os.makedirs(dataType)
    write_list(wavscp, os.path.join(dataType, 'wav.scp'))
    write_dict(spk2utt, os.path.join(dataType, 'spk2utt'))
    write_dict(utt2spk, os.path.join(dataType, 'utt2spk'))


if __name__ == '__main__':
    dataType = 'data_ESC-10'
    dataPath = '/home/lj/work/project/dataset/data_ESC-10'
    metaFile = '../src_esc10/files/meta.audiolist'

    make_files(metaFile, dataPath, dataType)
