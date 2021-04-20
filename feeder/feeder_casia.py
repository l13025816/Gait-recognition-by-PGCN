# sys
import os
import sys
import numpy as np
import random
import pickle
import json
# torch
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# operation
from . import tools

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def MaxMinNormalization(data):
    Max=np.max(data)
    Min=np.min(data)
    data = (data - Min) / (Max - Min)
    return data

class Feeder_casia(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition in kinetics-skeleton dataset
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        random_move: If true, perform randomly but continuously changed transformation to input sequence
        window_size: The length of the output sequence
        pose_matching: If ture, match the pose between two frames
        num_person_in: The number of people the feeder can observe in the input sequence
        num_person_out: The number of people the feeder in the output sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 data_path1,
                 ignore_empty_sample=True,
                 random_choose=False,
                 random_shift=False,
                 random_move=False,
                 window_size=-1,
                 pose_matching=False,
                 num_person_in=1,
                 num_person_out=1,
                 debug=False,
                 ):
        self.debug = debug
        self.data_path = data_path
        self.data_path1 = data_path1
        #self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.num_person_in = num_person_in
        self.num_person_out = num_person_out
        self.pose_matching = pose_matching
        self.ignore_empty_sample = ignore_empty_sample

        self.load_data()

    def load_data(self):
        # load file list
        self.sample_name = os.listdir(self.data_path)
        self.sample_name.sort()
        self.sample_name1 = os.listdir(self.data_path1)
        self.sample_name1.sort()
        # if self.debug:
        #     self.sample_name = self.sample_name[0:2]
        samplenum=0
        label=[]
        self.pair=[]
        #name:gallery, name1:probe
        for name1 in self.sample_name1:
             t_name1=name1.split('-')
             for name in self.sample_name:
                samplenum=samplenum+1
                t_name=name.split('-')
                #print(name1,t_name1,name,name1)
                if int(t_name[0])== int(t_name1[0]):
                    self.pair.append((name,name1))
                    label.append(1)
                else:
                    self.pair.append((name,name1))
                    label.append(0)
        label=np.array(label)
        self.label=label
        print(samplenum,len(label))
        self.N = samplenum#len(self.sample_name)  #sample
        self.C = 3  #channel
        self.T = self.window_size  #frame
        self.V = 18  #joint
        self.M = self.num_person_out  #person

    def __len__(self):
        return len(self.sample_name)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # output shape (C, T, V, M)
        # get data
        (sample_name,sample_name1) = self.pair[index]
        sample_path = os.path.join(self.data_path, sample_name)
        sample_path1 = os.path.join(self.data_path1, sample_name1)
        #gallery
        parents = os.listdir(sample_path)
        parents.sort()
        count=0
        video_info=[]
        for parent in parents:
            child = os.path.join(sample_path,parent)
            count += 1
            if count > self.window_size:
                break
            with open(child,"r") as f:
                s = json.load(f)
                if s["people"]==[]:
                    continue
                elif s["people"][0]==[]:
                    continue
                s_p=s["people"][0]["pose_keypoints_2d"]
                ss=np.array(s_p)
                ss=np.where(ss==0)
                if len(ss[0])>50:
                    continue
                H = (s_p[8*3+1]+s_p[11*3+1])/2-s_p[1*3+1]+1e-3
                #print(H)
                if H<10 or H>200:
                    continue
                xcenter=s_p[1*3]
                ycenter=s_p[1*3+1]
                zcenter=s_p[1*3+2]
                for i  in range(18):
                    s_p[i*3]=s_p[i*3]-xcenter
                    s_p[i*3+1]=s_p[i*3+1]-ycenter
                    s_p[i*3+2]=0
                s_p=np.array(s_p)
                s_p[abs(s_p>200)]=0
                s_p=s_p/H
                s_p=list(s_p)
                video_info.append(s_p)

        parents = os.listdir(sample_path1)
        parents.sort()

        #probe
        count=0
        video_info1=[]
        for parent in parents:
            child = os.path.join(sample_path1,parent)
            count += 1
            if count > self.window_size:
                break
            with open(child,"r") as f:
                s = json.load(f)
                if s["people"]==[]:
                    continue
                elif s["people"][0]==[]:
                    continue
                s_p=s["people"][0]["pose_keypoints_2d"]
                ss=np.array(s_p)
                ss=np.where(ss==0)
                if len(ss[0])>50:
                    continue
                H = (s_p[8*3+1]+s_p[11*3+1])/2-s_p[1*3+1]+1e-3
                if H<10 or H>200:
                    continue
                xcenter=s_p[1*3]
                ycenter=s_p[1*3+1]
                zcenter=s_p[1*3+2]
                for i  in range(18):
                    s_p[i*3]=s_p[i*3]-xcenter
                    s_p[i*3+1]=s_p[i*3+1]-ycenter
                    s_p[i*3+2]=0#s_p[i*3+2]-zcenter
                s_p=np.array(s_p)
                s_p[abs(s_p>200)]=0
                s_p=s_p/H
                s_p=list(s_p)
                video_info1.append(s_p)
        #print(len( video_info),len( video_info1))
        # fill data_numpy 3*260*18*1
        data_numpy = np.zeros((self.C, self.T*2, self.V, self.num_person_in))
        frame_index=0
        m=0
        for frame_info in video_info:
            frame_info = np.array(frame_info)
            for i in range(self.V):
                data_numpy[0, frame_index, i, m] = frame_info[i*3]
                data_numpy[1, frame_index, i, m] = frame_info[i*3+1]
                data_numpy[2, frame_index, i, m] = frame_info[i*3+2]
            frame_index+=1

        frame_index=self.T
        for frame_info in video_info1:
            frame_info = np.array(frame_info)
            for i in range(self.V):
                data_numpy[0, frame_index, i, m] = frame_info[i*3]
                data_numpy[1, frame_index, i, m] = frame_info[i*3+1]
                data_numpy[2, frame_index, i, m] = frame_info[i*3+2]
            frame_index+=1
        label=int(self.label[index])

        # data augmentation
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        # match poses between 2 frames
        if self.pose_matching:
            data_numpy = tools.openpose_match(data_numpy)
        return data_numpy, label

    def top_k(self, score, top_k):
        assert (all(self.label >= 0))

        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def top_k_by_category(self, score, top_k):
        assert (all(self.label >= 0))
        return tools.top_k_by_category(self.label, score, top_k)

    def calculate_recall_precision(self, score):
        assert (all(self.label >= 0))
        return tools.calculate_recall_precision(self.label, score)
