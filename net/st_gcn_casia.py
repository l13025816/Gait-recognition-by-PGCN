import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph
import numpy as np


class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()
        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        # build networks
        spatial_kernel_size = A.size(0) #unifor 1, distance 2, spatial 3
        temporal_kernel_size = 3
        kernel_size = (temporal_kernel_size, spatial_kernel_size) #9,3
        #BN for single stream
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1)*100)
        #BN for concat stream
        self.data_bn2 = nn.BatchNorm1d(in_channels*2 * A.size(1)*100)# 6* 18

        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            #3 64 (9,3)
            st_gcn(in_channels, 16, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(16, 16, kernel_size, 1, **kwargs),
            st_gcn(16, 64, kernel_size, 2, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 64, kernel_size, 2, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 16, kernel_size, 2, **kwargs),
            st_gcn(16, 16, kernel_size, 1, **kwargs),
        ))

        self.st_gcn_networks2 = nn.ModuleList((
            #3 64 (9,3)
            st_gcn(in_channels*2, 16, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(16, 16, kernel_size, 1, **kwargs),
            st_gcn(16, 64, kernel_size, 2, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 64, kernel_size, 2, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 16, kernel_size, 2, **kwargs),
            st_gcn(16, 16, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        #train.yaml True
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        if edge_importance_weighting:
            self.edge_importance2 = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks2
            ])
        else:
            self.edge_importance2 = [1] * len(self.st_gcn_networks2)

        self.fuse= nn.ParameterList([
                nn.Parameter(torch.ones(1))
                for i in range(2)
            ])
        # fcn for prediction
        self.FCN= nn.ModuleList((
          nn.Linear(16*18, 128),
          nn.BatchNorm1d(128),
          nn.Linear(128,64),
          nn.BatchNorm1d(64),
          nn.Linear(64, 32),
          nn.BatchNorm1d(32),
          nn.Linear(32, 1)
          ))

        self.FCN2= nn.ModuleList((
          nn.Linear(16*18, 128),
          nn.BatchNorm1d(128),
          nn.Linear(128,64),
          nn.BatchNorm1d(64),
          nn.Linear(64, 32),
          nn.BatchNorm1d(32),
          nn.Linear(32, 1)
          ))

        #self.sig=torch.tanh()
        #self.fcn = nn.Conv2d(1000, num_class, kernel_size=1)
        #self.soft=nn.Softmax(dim = 1)

    def forward(self, x):
        xf=x[:,:,0:100,:,:]
        xb=x[:,:,130:230,:,:]
        x=torch.cat((xf,xb),1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C* T)
        x = self.data_bn2(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks2, self.edge_importance2):
            x, _ = gcn(x, self.A * importance)
        #64 16 7 18
        x = F.avg_pool2d(x, (x.size()[2],1))
        x = x.view(x.size(0), -1)
        for fcn in (self.FCN2):
            x = fcn(x)
        feature1=x

        #first part
        x=xf
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C * T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        x = F.avg_pool2d(x, (x.size()[2],1))
        x = x.view(x.size(0), -1)
        featurexf = x

        x=xb
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C* T)
        x = self.data_bn(x)
        #x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        x = F.avg_pool2d(x, (x.size()[2],1))
        x = x.view(x.size(0), -1)
        featurexb = x

        feature=torch.abs(featurexf-featurexb)
        #feature=torch.cat((featurexf,featurexb),1)
        for fcn in (self.FCN):
            feature = fcn(feature)
        feature2=feature
        #featureall=torch.tanh(feature1)+1-torch.tanh(feature2)
        #print(self.fuse[0],self.fuse[1])
        featureall=self.fuse[0]*feature1+self.fuse[1]*feature2
        return featureall, feature1, feature2, featurexf, featurexb


    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        #1,3,50,18,1
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        #1,1,18,3,50
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        #1,1,3,50,18
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        print(c,t,v)
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature

class st_gcn(nn.Module):
    """Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0.2,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            #nn.MaxPool2d(kernel_size=(stride, 1)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    #kernel_size=(kernel_size[0], 1),
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)
        #self.edge_importance = nn.Parameter(torch.ones((3,18,18)))


    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A


