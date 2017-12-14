import os
import sys
import logging

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