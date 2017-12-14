import sys
sys.path.append('../')
from util import *

def extract_feature(y, sr):
    #melspec = librosa.feature.melspectrogram(signal, sr=22050, n_fft=1024, hop_length=512, n_mels=BANDS)
    #logspec = librosa.logamplitude(melspec)

    eps = numpy.spacing(1)
    power_spectrogram = numpy.abs(librosa.stft(
        y + eps, n_fft=nfft, win_length=wlen, hop_length=hlen, center=False))**2
    mel_basis = librosa.filters.mel(sr=sr, n_fft=nfft, n_mels=nmel)
    mel_spect = numpy.dot(mel_basis, power_spectrogram)
    log_spect = librosa.logamplitude(mel_spect)
    #print 'log_spec: ', log_spect.shape
    return log_spect

if __name__ == '__main__':
    print '------------------------------------------------------------'
    sr = 22050
    nfft = int(0.025*sr)
    wlen = int(0.025*sr)
    hlen = int(0.010*sr)
    nmel = 64

    audioPath = '/home/lj/work/project/dataset/data_ESC-10'
    check_path(audioPath)

    featPath = '../feature/feature_ESC-10/{:d}_logspec'.format(sr)
    make_path(featPath)

    metaFile = 'files/meta.audiolist'
    check_file(metaFile)

    lines = open(metaFile, 'rb').readlines()
    for idx, items in enumerate(lines):
        items = items.split('\n')[0].split('\t')
        audioFile = os.path.join(audioPath, items[0])
        audioName = os.path.basename(items[0])
        audioLabs = int(items[1])

        check_file(audioFile)
        y, sr = librosa.load(audioFile, sr=sr, mono=True)
        y_ = y/y.max()
        data = extract_feature(y_, sr)
        #plot_wave(y_, sr=sr, logspec=data, mono=True, audioName=audioFile)
        pickle.dump(data, open(os.path.join(featPath, audioName.replace('.ogg', '.fea')), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)
        print idx+1, audioFile, audioName, audioLabs, data.shape

        #break



