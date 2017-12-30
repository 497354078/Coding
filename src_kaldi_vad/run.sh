#!/bin/bash
# Copyright 2015-2017   David Snyder
#                2015   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2015   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (EERs) are inline in comments below.



. cmd.sh
. path.sh
set -e
dataType=ESC-10
metaFile=../src_esc10/files/meta.audiolist
dataPath=/home/lj/work/project/dataset/data_$dataType
destPath=/home/lj/work/project/dataset/data_audio
dataDir=`pwd`/data_$dataType

mfccdir=`pwd`/mfcc_$dataType
vaddir=`pwd`/vad_$dataType

echo $dataType
echo $metaFile
echo $dataPath
echo $dataDir
echo $mfccdir
echo $vaddir

#python make_kaldi_files.py $dataDir $dataPath $destPath $metaFile

file=$dataDir/wav.scp
length=`cat $file | wc -l`
echo $file 
echo $length
index=1
while(($index<=$length))
do
    audioFile=`awk '{print $2}' $file | head -n $index | tail -n -1`
    audioName=${audioFile##*/}
    dirName=${audioFile%/*}
    echo ${dirName}/${audioName}
    let "index++"
    destFile=${audioFile//ogg/wav}
    echo $destFile
    sox $audioFile $destFile
    #scale=`sox $audioFile -n stat -v 2>&1`
    #scale=`echo "$scale*0.9"|bc`
    #sox -v $scale $audioFile -r 16000 -b 16 -c 1 $destFile
    break
done

: '
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 4 --cmd "$train_cmd" \
  $dataDir exp/make_mfcc $mfccdir

sid/compute_vad_decision.sh --nj 4 --cmd "$train_cmd" \
  $dataDir exp/make_vad $vaddir
'
