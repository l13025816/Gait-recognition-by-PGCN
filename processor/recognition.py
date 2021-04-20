#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import itertools
import matplotlib as mpl
#train set
train_num=62

class TripletLoss_out(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss_out, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)
    def forward(self, inputs, targets):
        inputs=torch.squeeze(inputs)
        targets=torch.squeeze(targets)
        #print(inputs,targets)
        dist_ap, dist_an = [], []
        a=inputs[targets==1]
        b=inputs[targets==0]
        #print(a.size(0),b.size(0))
        for i in range(a.size(0)):
            for j in range(b.size(0)):
                dist_ap.append(a[i].unsqueeze(0))
                dist_an.append(b[j].unsqueeze(0))

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        m=nn.Sigmoid()
        dist_ap=m(dist_ap)
        dist_an=m(dist_an)
        #print(dist_ap,dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_ap, dist_an, y)

class TripletLoss_dist(nn.Module):
    def __init__(self, margin=5):
        super(TripletLoss_dist, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)
    def forward(self, recon_x, x, targets):
        inputs=torch.squeeze(torch.sqrt(torch.sum(torch.pow(recon_x-x,2),dim=1))+1e-3)
        targets=torch.squeeze(targets)
        dist_ap, dist_an = [], []
        a=inputs[targets==1]
        b=inputs[targets==0]
        #print(a,b)
        for i in range(a.size(0)):
            for j in range(b.size(0)):
                dist_ap.append(a[i].unsqueeze(0))
                dist_an.append(b[j].unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # m=nn.Sigmoid()
        # dist_ap=m(dist_ap)
        # dist_an=m(dist_an)
        #print(dis print(ere)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)

class TripletLoss_sim(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss_sim, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)
    def forward(self, recon_x, x, targets):
        inputs=torch.squeeze(torch.cosine_similarity(recon_x, x, dim=1))
        targets=torch.squeeze(targets)
        dist_ap, dist_an = [], []
        a=inputs[targets==1]
        b=inputs[targets==0]
        for i in range(a.size(0)):
            for j in range(b.size(0)):
                dist_ap.append(a[i].unsqueeze(0))
                dist_an.append(b[j].unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        #print(dist_ap,dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_ap, dist_an, y)

def weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.1)
        m.bias.data.fill_(0)

class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        #self.loss = nn.CrossEntropyLoss()
        self.loss3=nn.CosineEmbeddingLoss(margin=0.3)
        #self.loss4=nn.HingeEmbeddingLoss()
        #self.loss5=nn.TripletMarginLoss(margin=0.3, p=2)
        self.loss6=nn.BCEWithLogitsLoss()
        #self.soft=nn.Softmax(dim = 1)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            # for param_group in self.optimizer.param_groups:
            #     param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        self.result[:,0:train_num+1]=0
        rank = self.result.argsort()
        #for i, l in enumerate(self.label):
        #    print(l,rank[i, -1:])
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def genmask(self,data,rate):
        mask=torch.ones(data.shape)
        num2=int(data.shape[2]*rate)
        num3=int(data.shape[3]*rate)
        k2=np.random.randint(0,data.shape[2],num2)
        k3=np.random.randint(0,data.shape[3],num3)
        #print(mask.shape)
        for i in k3:
            mask[:,:,k2,i,0]=0
        #mask[:,:,:,k3,0]=0
        data = data.masked_fill(mask = mask.bool(), value=torch.tensor(0))
        return data

    def disloss(self,output2,output3,label):
        loss1=torch.sum(label*torch.norm(output2-output3,p=2))
        loss2=torch.sum((1-label)*1/torch.norm(output2-output3,p=2))
        return loss1+loss2


    def gendata(self,data,rate):
        #64,3,50,18,1

        num0=int(data.shape[0]*rate)
        num2=int(data.shape[2]*rate)
        num3=int(data.shape[3]*rate)
        k0=np.random.randint(0,data.shape[0],num0)
        #k2=np.random.randint(0,data.shape[2],num2)
        #k3=np.random.randint(0,data.shape[3],num3)

        # k1=[0,0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4,1.5]
        # k2=[0,0.7,0.8, 0.9, 1.0, 1.1, 1.2,1.3]
        k1=[-3,-2,-1,0,1,2,3]
        k2=[-3,-2,-1,0,1,2,3]
        k3=np.random.randint(0,len(k1),18)
        k4=np.random.randint(0,len(k2),18)
        #print(k3,k4)

        #print(k4)
        #print(mask.shape)
        for i0 in k0:#range(data.shape[0]):
            for j0 in range(18):
                data[i0,0,:,j0,0]=data[i0,0,:,j0,0]+k1[k3[j0]]
                data[i0,1,:,j0,0]=data[i0,1,:,j0,0]+k2[k4[j0]]

        #data = data.masked_fill(mask = mask.bool(), value=torch.tensor(0))
        return data


    # Define the contrastive loss function, NT_Xent
    def NT_Xent(zi, zj, tau=1):
        """ Calculates the contrastive loss of the input data using NT_Xent. The
        equation can be found in the paper: https://arxiv.org/pdf/2002.05709.pdf

        Args:
            zi: One half of the input data, shape = (batch_size, feature_1, feature_2, ..., feature_N)
            zj: Other half of the input data, must have the same shape as zi
            tau: Temperature parameter (a constant), default = 1.

        Returns:
            loss: The complete NT_Xent constrastive loss
        """
        z = np.concatenate((zi, zj), 0)
        loss = 0
        for k in range(zi.shape[0]):
            # Numerator (compare i,j & j,i)
            i = k
            j = k + zi.shape[0]
            sim_ij = np.squeeze(cosine_similarity(z[i].reshape(1, -1), z[j].reshape(1, -1)))
            sim_ji = np.squeeze(cosine_similarity(z[j].reshape(1, -1), z[i].reshape(1, -1)))
            numerator_ij = np.exp(sim_ij / tau)
            numerator_ji = np.exp(sim_ji / tau)

            # Denominator (compare i & j to all samples apart from themselves)
            sim_ik = np.squeeze(cosine_similarity(z[i].reshape(1, -1), z[np.arange(z.shape[0]) != i]))
            sim_jk = np.squeeze(cosine_similarity(z[j].reshape(1, -1), z[np.arange(z.shape[0]) != j]))
            denominator_ik = np.sum(np.exp(sim_ik / tau))
            denominator_jk = np.sum(np.exp(sim_jk / tau))

            # Calculate individual and combined losses
            loss_ij = - np.log(numerator_ij / denominator_ik)
            loss_ji = - np.log(numerator_ji / denominator_jk)
            loss += loss_ij + loss_ji

        # Divide by the total number of samples
        loss /= z.shape[0]

        return loss

    def loss2(self,recon_x, x, y):
        margin=0.3
        loss=np.squeeze(torch.cosine_similarity(recon_x, x, dim=1))
        dist=np.squeeze(torch.sqrt(torch.sum(torch.pow(recon_x-x,2),dim=1))+1e-3)

        y=np.squeeze(y)
        a=loss[y==1]
        loss1=1-a
        b=loss[y==0]
        zero=torch.zeros_like(b)
        loss2=torch.max(zero,b)

        da=dist[y==1]
        db=dist[y==0]
        print(da,db)
        #zero1=torch.zeros_like(db)
        da=torch.tanh(da)
        db=1-torch.tanh(db)

        return torch.sum(loss1)/a.size(0)+torch.sum(loss2)/b.size(0) \
               , torch.sum(da)/da.size(0)+torch.sum(db)/db.size(0)

    def loss7(self,x1,x2, target):
        #def CustomCosineEmbeddingLoss(x1, x2, target):
        x1_ = torch.sqrt(torch.sum(x1 * x1, dim = 1)) # |x1|
        x2_ = torch.sqrt(torch.sum(x2 * x2, dim = 1)) # |x2|
        cos_x1_x2 = torch.sum(x1 * x2, dim = 1)/(x1_ * x2_)
        ans = torch.mean(target- cos_x1_x2)
        return ans

    def extract2(self,data,label):

        # xf=data[:,:,0:50,:,:]
        # xb=data[:,:,50:100,:,:]

        # bat=xf.size(0)
        # bat1=xf.size()
        # xf = xf.reshape(bat, -1)
        # mean = xf.mean(dim=1).reshape(bat,-1)
        # std = xf.std(dim=1, unbiased=False).reshape(bat,-1)
        # xf = (xf - mean)/(std+1e-5)
        # xf=xf.reshape(bat1)

        # bat=xb.size(0)
        # bat1=xb.size()
        # xb = xb.reshape(bat, -1)
        # mean = xb.mean(dim=1).reshape(bat,-1)
        # std = xb.std(dim=1, unbiased=False).reshape(bat,-1)
        # xb = (xb - mean)/(std+1e-5)
        # xb=xb.reshape(bat1)

        # data=torch.cat((xf,xb),1)
        #100,3,100,18,1

        data = data.contiguous().view(data.size(0), -1)
        #label = label.contiguous().view(label.size(0), -1)

        a=data[label.gt(0)]
        label=label*-1+1
        b=data[label.gt(0)]

        data=data.data.cpu().numpy()
        a=a.data.cpu().numpy()
        b=b.data.cpu().numpy()


        print(data.shape)

        fig = plt.figure()
        pca = PCA(n_components=3)
        pca.fit(data)
        #print pca.explained_variance_ratio_
        #print pca.explained_variance_
        ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
        X = pca.transform(a)
        #print(X)
        ax.scatter(X[:, 0],X[:, 1],X[:, 2], marker='*',c='red',s=40)
        #ax.scatter(X[:, 0],X[:, 1].reshape(1.-1),X[:, 2].reshape(1.-1), marker='*',c='red',s=30)
        X = pca.transform(b)
        #print(X)
        ax.scatter(X[:, 0],X[:, 1],X[:, 2],s=40, marker='*',c='g')
        plt.show()
        print(ere)

    def extract(self,xf,xb):

        xf = xf.contiguous().view(xf.size(0), -1)
        xb = xb.contiguous().view(xb.size(0), -1)

        xf=xf.data.cpu().numpy()
        xb=xb.data.cpu().numpy()
        print(np.max(xf),np.max(xb))
        print(xf.shape,xb.shape)
        #xf=xf.astype(np.float)
        #print(ere)
        temp=[]
        labelx=[]
        number=20
        for i in range(number):
              #temp=recon_y2[0*kk:1*kk,:]
            tt=xf[i*10:(i+1)*10,:]
            #tt=tt.reshape((1,-1))
            temp.append(tt)
            labelx.append(i+1)

        temp111=temp[0]
        for i in range(1,number):
            temp111=np.vstack((temp111,temp[i]))

        print(temp111.shape)

        fig = plt.figure()
        # tsne=TSNE(n_components=3)
        # Y=tsne.fit_transform(temp111)
        pca = PCA(n_components=3)
        pca.fit(temp111)
        print (pca.explained_variance_ratio_)
        print (pca.explained_variance_)
        ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)

        labelx=np.array(labelx)
        print(labelx)
        #print(X.shape)
        for i in range(number):
            #X = Y[i*10:(i+1)*10,:]
            print(labelx[i])
            X = pca.transform(temp[i])
            #cmap = mpl.cm.get_cmap('Set1', labelx.shape[0])
            #colorst = cmap(np.linspace(0, 1, labelx.shape[0]))
            ax.scatter(X[:, 0],X[:, 1],X[:, 2],s=40, marker='*', label=labelx[i])

        colormap=plt.cm.gist_ncar
        colorst=[colormap(i) for i in np.linspace(0,1,len(ax.lines))]
       # print(ax.collections)
        for t,j1 in enumerate(ax.lines):
            #print(t,colorst[t])
            j1.set_color(colorst[t])
        plt.show()
        print(ere)

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        trip=TripletLoss_out()
        trip1=TripletLoss_sim()
        trip2=TripletLoss_dist()
        mm=0
        for data, label in loader:
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)
            #data[abs(data)>50]=0
            data[data!=data]=0
            #y1=label.float()*2-1
            # xf=data[:,:,0:100,:,:]
            # xb=data[:,:,130:230,:,:]
            output,output1,output2,output3,output4 = self.model(data)
            label= label.view(label.size(0), -1)
            y2=label.float()*(-1)+1
            # output = np.squeeze(output)
            # label = np.squeeze(label)
            # output1 = output1.view(output.size(0), -1)
            # output2 = output2.view(output.size(0), -1)

            # if mm==0:
            #     data_show=output3
            # else:
            #     data_show=torch.cat((data_show,output3),0)
            # mm=mm+1
            # if mm==6:
            #     print(data_show.size())
            #     self.extract(data_show,data_show)
            #     print(ere)
            # continue
            #loss5=self.loss3(output3,output4, y1)
            loss3=trip.forward(output1,label)
            loss4=trip.forward(output2,label)
            #loss = self.loss6(output, label.float())
            loss1 = self.loss6(output1, label.float())
            loss2 = self.loss6(output2, label.float())
            #loss7=self.loss7(output, label)
            loss5=trip1.forward(output3,output4, label.float())
            loss6=trip2.forward(output3,output4, label.float())
            #print (loss1.requires_grad,loss3.requires_grad)
            #print (loss2.requires_grad,loss4.requires_grad)
            #print (loss5.requires_grad,loss6.requires_grad)
            #print(torch.cosine_similarity(output3,output4, dim=1),label)
            #print(loss8)
            # loss = self.loss(output,label)
            # loss1 = self.loss(output1,label)
            # loss2 = self.loss(output2,label)-
            #print(loss1,loss2,loss3,loss4,loss5,loss6)
            #print(output,output1,output2)
            #loss=loss1+loss2+loss3+loss4+loss5+loss6
            #print(loss5,loss6)
            loss=loss1+loss2+loss3+loss4+loss5+loss6
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []
        probeid=0
        correct=0

        for data, label in loader:
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)
            #data[abs(data>50)]=0
            data[data!=data]=0
            # xf=data[:,:,0:100,:,:]
            # xb=data[:,:,130:230,:,:]
            # xf = xf.contiguous().view(xf.size(0), -1)
            # xb = xb.contiguous().view(xb.size(0), -1)
            # print(xf,xb,label)
            #print(ere)
            # inference
            with torch.no_grad():
                output,output1,output2,output3,output4 = self.model(data)
                #output = self.model(data)
            # output3 = output3.contiguous().view(output3.size(0), -1)
            # output4 = output4.contiguous().view(output4.size(0), -1)
            # print(output3,output4,label)
            #self.extract(output3,output4)
            label= label.view(label.size(0), -1)
            probe1=torch.cosine_similarity(output3, output4, dim=1)
            #k=nn.Sigmoid()
            #print(k(output),label)
            probe=output.data.cpu().numpy()
            probe1=probe1.data.cpu().numpy()
            probe2=output1.data.cpu().numpy()
            probe3=output2.data.cpu().numpy()
            #probe=probe[:,1]
            # probe=np.reshape(probe,(int(probe.shape[0]/4),-1))
            # probe=np.mean(probe,1)
            #print(probe,probe1,label)
            #print(probe1)
            probe=np.argmax(probe)
            probe1=np.argmax(probe1)
            probe2=np.argmax(probe2)
            probe3=np.argmax(probe3)
            #print(probe1[probe])
            #result_frag.append(output.data.cpu().numpy())
            print(probeid,int(np.floor(probeid/2)),int(np.floor(probe/4)),int(np.floor(probe1/4)),int(np.floor(probe1/4)),int(np.floor(probe2/4)))
            if int(np.floor(probeid/2))==int(np.floor(probe/4)):
                correct=correct+1
            # else:
            #     print(rawprobe)

            probeid+=1

        print(correct,124,correct/124)

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser



#python main.py recognition --phase train -c /home/bird/xuke/st-gcn-master/config/st_gcn/casia-skeleton/train.yaml
#python main.py recognition  --phase test -c /home/bird/xuke/st-gcn-master/config/st_gcn/casia-skeleton/test.yaml