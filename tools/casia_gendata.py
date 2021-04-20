# -*- coding: utf-8 -*-
import os
import sys
import pickle
import argparse

import numpy as np
from numpy.lib.format import open_memmap

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from feeder.feeder_casia import Feeder_casia

toolbar_width = 30
def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')

def end_toolbar():
    sys.stdout.write("\n")

def gendata(
        data_path,
        data_path1,
        data_out_path,
        label_out_path,
        num_person_in=1,  #observe the first 5 persons
        num_person_out=1,  #then choose 2 persons with the highest score
        max_frame=130):

    feeder = Feeder_casia(
        data_path=data_path,
        data_path1=data_path1,
        num_person_in=num_person_in,
        num_person_out=num_person_out,
        window_size=max_frame,
        )

    sample_name = feeder.pair
    sample_label = []

    fp = open_memmap(
        data_out_path,
        dtype='float32',
        mode='w+',
        shape=(len(sample_name), 3, max_frame*2, 18, num_person_out))

    for i, s in enumerate(sample_name):
        data, label = feeder[i]
        print_toolbar(i * 1.0 / len(sample_name),
                      '({:>5}/{:<5}) Processing data: '.format(
                          i + 1, len(sample_name)))
        fp[i, :, 0:data.shape[1], :, :] = data
        sample_label.append(label)

    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

if __name__ == '__main__':
    #part = ['train74-all', 'val74-180nm','val74-180bg','val74-180cl']
    out_folder='/home/bird/disk2/mydisk1104/xuke/skeleton-data/casia-skeleton'
    #angle='018' '018','036','054','072','090','108','126','144','162','180'
    for angle in ['000']:#gallery view
        angle1='000' #probe view
        #***test set
        part = ['train62-{}'.format(angle),'probe62-{}nm'.format(angle),'probe62-{}bg'.format(angle),'probe62-{}cl'.format(angle)]
        #***train set
        #part=['train62-{}'.format(angle)]
        for p in part:
            #data_path = '{}/casia_{}'.format(arg.data_path, p)
            if p=='train62-{}'.format(angle):
                p1='train62-{}'.format(angle1)
                data_path ='/home/bird/xuke/st-gcn-master/data/casia/{}'.format(p)
                data_path1 ='/home/bird/xuke/st-gcn-master/data/casia/{}'.format(p1)
            elif p=='probe62-{}nm'.format(angle):
                p1='probe62-{}nm'.format(angle1)
                data_path ='/home/bird/xuke/st-gcn-master/data/casia/gallery62-{}'.format(angle)
                data_path1 ='/home/bird/xuke/st-gcn-master/data/casia/{}'.format(p1)
            elif p=='probe62-{}bg'.format(angle):
                p1='probe62-{}bg'.format(angle1)
                data_path ='/home/bird/xuke/st-gcn-master/data/casia/gallery62-{}'.format(angle)
                data_path1 ='/home/bird/xuke/st-gcn-master/data/casia/{}'.format(p1)
            elif p=='probe62-{}cl'.format(angle):
                p1='probe62-{}cl'.format(angle1)
                data_path ='/home/bird/xuke/st-gcn-master/data/casia/gallery62-{}'.format(angle)
                data_path1 ='/home/bird/xuke/st-gcn-master/data/casia/{}'.format(p1)
            label_path = ''
            data_out_path = '{}/{}-{}_data.npy'.format(out_folder, p, angle1)
            label_out_path = '{}/{}-{}_label.pkl'.format(out_folder, p, angle1)

            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            gendata(data_path, data_path1, data_out_path, label_out_path)

        #python ./tools/casia_gendata.py