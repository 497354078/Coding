import os
import sys
import numpy as np
import cPickle as pickle

lines = open('files/ESC-10-silent', 'rb').readlines()
vad2list = {}
for items in lines:
    items = items.split('\n')[0].split('\t')
    #print items
    audioName = items[0]
    stime = float(items[1])
    etime = float(items[2])

    stime = int(max(0, stime-0.05) / 0.010)
    etime = int(min(etime+0.05, 5) / 0.010)

    if audioName not in vad2list:
        vad2list[audioName] = []
    vad2list[audioName].append((stime, etime))

vad2numpy = {}
for audioName in vad2list:
    mark = np.zeros(510)
    for stime, etime in vad2list[audioName]:
        while stime <= etime:
            mark[stime] = 1
            stime += 1
    vad2numpy[audioName] = mark

print 'vad2list: ', len(vad2list)
print 'vad2numpy: ', len(vad2numpy)

pickle.dump(vad2list, open('files/vad2list.dict', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(vad2numpy, open('files/vad2numpy.dict', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)






