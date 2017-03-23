#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import requests
import argparse
import progressbar
import numpy as np
import os.path as osp
from astropy import wcs
from astropy.io import fits
from bs4 import BeautifulSoup

parser = argparse.ArgumentParser(description='Retrieve labels of astronomical '
                                 'objects based on their FK5 coordinates')

parser.add_argument('path', metavar='path',
                    help='Path that contains FITS images of observations')

parser.add_argument('--prefix', default='diff-*.fits',
                    help='Regex that matches the filenames to be labeled')


URL = 'http://simbad.u-strasbg.fr/simbad/sim-coo'
params = {
    'Coord': '215.0939663516772 52.335801868486165',
    'Radius': 2,
    'Radius.unit': 'arcmin'
}


def retrieve_label(ra, dec):
    params['Coord'] = '{0} {1}'.format(ra, dec)
    soup = BeautifulSoup(requests.get(URL, params=params).text, 'lxml')
    tab = soup.find('td', attrs={'id': 'basic_data'})
    try:
        label = tab.find_all('td')[0].find('font').find('b').next.next
        label = label.split('\n')[2]
    except AttributeError:
        label = 'Other'
    print(label)
    return label


def process_labels(path, prefix):
    sequences = {}
    files = glob.glob(osp.join(path, prefix))
    bar = progressbar.ProgressBar(redirect_stdout=True)
    for file in bar(files):
        print(file)
        filename = osp.basename(file)
        seq = filename.split('-')[1]
        if seq not in sequences:
            hdulist = fits.open(file)
            wcs_conv = wcs.WCS(hdulist[1])
            ra, dec = wcs_conv.wcs_pix2world(0, 0, True)
            print(ra, dec)
            sequences[seq] = retrieve_label(ra, dec)

    with open('labels.csv', 'r') as fp:
        fp.write('obj,type\n')
        for obj in sequences:
            fp.write('{0},{1}\n'.format(obj, sequences[obj]))


if __name__ == '__main__':
    args = parser.parse_args()
    process_labels(args.path, args.prefix)
