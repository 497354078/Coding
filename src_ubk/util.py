import os
import sys
import logging


classid2name = {
    0 : 'air_conditioner',
    1 : 'car_horn',
    2 : 'children_playing',
    3 : 'dog_bark',
    4 : 'drilling',
    5 : 'engine_idling',
    6 : 'gun_shot',
    7 : 'jackhammer',
    8 : 'siren',
    9 : 'street_music',
    }


def rePrint(printStr):
    print printStr
    logging.info(printStr)


def check_path(path):
    if not os.path.exists(path):
        raise IOError('cannot found path: {:s}'.format(path))

def check_file(files):
    if not os.path.isfile(files):
        raise IOError('cannot found file: {:s}'.format(files))

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)