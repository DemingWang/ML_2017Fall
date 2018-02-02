#!python
# -*- coding: utf-8 -*-#
"""
Download files from websites.

@author: Bhishan Poudel

@date: Oct 15, 2017

@email: bhishanpdl@gmail.com

"""
# Imports
import urllib.parse as urlparse
import wget
import glob, os, subprocess
from os.path import splitext, basename

def download():

    urls = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
            ]

    basenames = [os.path.basename(urlparse.urlparse(url).path) for url in urls]
    # print("basenames = {}".format(basenames))
    
    for url in urls:
        try:
            basename = os.path.basename(urlparse.urlparse(url).path)
            if not os.path.isfile(basename):
                print("Downloading: {}".format(basename))
                wget.download(url)
        except:
            print("Could not download the files.")
    
    if not os.path.isdir('data'):
        os.makedirs('data')
     
    for basename in basenames:
        if os.path.isfile(basename):
            print("gunzip : {}".format(basename))
            subprocess.call('mv {} data/'.format(basename),shell=True)
            subprocess.call('gunzip data/{}'.format(basename),shell=True)
            subprocess.call('rm -rf data/{}'.format(basename),shell=True)
            


def main():
    """Run main function."""
    download()

if __name__ == "__main__":
    main()
