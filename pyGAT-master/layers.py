import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
#from layers import GraphConvolution
import cupy as cp
import numpy as np
import torch.nn as nn
import torch
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from sklearn.cluster import KMeans

import scipy
import pdb
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,0'
torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print("device:",device)
def get_dc(data, m, size):
    data = data.to(device)
    y = torch.pdist(data.detach())
    z = squareform(cp.asarray(y).get())
    z = torch.Tensor(z)
    #y = np.array(y)
    # a = y.mean()
    t1, t2 = torch.topk(z, m + 1, largest=False) #通常该函数返回2个值，第一个值为排序的数组，第二个值为该数组中获取到的元素在原数组中的位置标号。
    t3 = t2
    t2 = data[t2]
    a = t1[:,1].mean()
    #print("hibog2_a:",a)
    return t2, a, t1,t3 #t2是最近邻几个点的表示，t1是最近邻几个点的距离

import cupy as cp
from cupy.linalg import norm

def get_center_1(data, cluster_num):
    #pdb.set_trace()
    #torch.set_printoptions(threshold=np.inf)
    #print("data:",data.size())
    #print(data)
    #data1 = cp.cpu().asarray(data).get()
    data1 = data.detach()
    #print("data1.device:",data1.device)
    ori_data = torch.pdist(data1)
    ori_data = ori_data.cpu().numpy()
    ori_data = squareform(ori_data)
    ori_data = torch.Tensor(ori_data)
    ori_data = ori_data.to(device)
    #print("ori_data_0.shape:",ori_data.size())
    #print("ori_data.device:",ori_data.device)
    #print(ori_data)
    #pdb.set_trace()
    m = 5
    _, dc, _, loc = get_dc(data, m, size=2)
    #pdb.set_trace()
    ori_data_tmp = ori_data.clone()
    #print("ori_data_tmp.shape:",ori_data_tmp.size())
    #print(ori_data_tmp)
    ori_data_tmp = ori_data_tmp.to(device)
    dc = dc.to(device)
    #print("ori_data_tmp.device:",ori_data_tmp.device)
    #print("dc.device:",dc.device)
    #print("loc.shape:",loc.size())
    #print(loc)
#with torch.autograd.profiler.profile(enabled=True, use_cuda=False) as prof:
    for j in range(0, loc.shape[0]):
        for i in range(1, cluster_num+1):
            ori_data[j][loc[j][i]] = 0
    #print("ori_data_1.shape:",ori_data.size())
    #print(ori_data)
    ori_data = ori_data_tmp - ori_data
    ori_data = ori_data.detach().cpu().numpy()
    ori_data = cp.array(ori_data)
    #print("ori_data2.device:",ori_data.device)
    #pdb.set_trace()
    #print("ori_data_2.type:",type(ori_data))
    #print("ori_data_2.dtype:",(ori_data).dtype)
    #print(ori_data)
    
    # 假设ori_data, dc, loc, cluster_num都是numpy数组
# 使用np.arange和np.reshape创建一个与ori_data形状相同的索引矩阵
    loc = np.asarray(loc)
    index = cp.arange(ori_data.size).reshape(ori_data.shape)
    #ori_data = ori_data.cpu().numpy()
# 使用np.take_along_axis从ori_data中取出每行的第loc[i][1]个元素，作为nearest_dis
    nearest_dis = cp.take_along_axis(ori_data, loc[:, 1:2], axis=1)

# 使用np.where判断ori_data中不为零的元素，并对它们进行计算
    ori_data = cp.where(ori_data != 0, dc * nearest_dis / (ori_data * ori_data), ori_data)

# 使用np.where判断每行中超过cluster_num个不为零的元素，并将它们设为零
    mask = index >= cp.argsort(ori_data, axis=1)[:, -cluster_num:].min(axis=1, keepdims=True)
    ori_data = cp.where(mask, 0, ori_data)
    '''
    for i in range(0, ori_data.shape[0]):
        flag=0
        nearest_dis = ori_data[i][loc[i][1]]
        print("nearest_dis.device:",nearest_dis.device)
        for j in range(0, ori_data.shape[1]):
            if ori_data[i][j] != 0:
                ori_data[i][j] = dc * nearest_dis / (ori_data[i][j] * ori_data[i][j])
                flag+=1
            if flag == cluster_num:
                break
    '''    
    ori_data = ori_data - 9e-15
    ori_data =torch.Tensor(ori_data)
    #print("ori_data.shape:",ori_data.size())
    #print(ori_data)
    #pdb.set_trace()
#print(prof.key_averages().table(sort_by="cpu_time_total"))
    kmeans = KMeans(
        n_clusters=3, init="k-means++"
    ).fit(data.detach().cpu().numpy())#cp.asarray(data).get())
    center = kmeans.cluster_centers_
    center1 = torch.tensor(center)
    data = data.to(device)
    #print("data:",data.size())
    #print(data)
    #print("center1:",center1.size())
    #print(center1)
    #pdb.set_trace()
    data1 = torch.cat((data.to(device), center1.to(device)), dim=0)
    y_3 = pdist(data.detach().cpu().numpy())#cp.asarray(center1).get())
    z_3 = squareform(y_3)
    z_3 = torch.zeros(center1.shape[0], center1.shape[0]) - 9e15
    n = data.shape[0]
    m = center1.shape[0]
    
    #data = data.reshape(n, 1, -1).repeat(m, axis=1)
    #center1 = center1.reshape(1, m, -1).repeat(n, axis=0)
    data = data.unsqueeze(1).expand(n, m, -1)  # 扩展data1的维度为(n, m, feature_dim1)
    center1 = center1.unsqueeze(0).expand(n, m, -1)  # 扩展data2的维度为(n, m, feature_dim2)
    #distances_1 = torch.norm(data.to(device) - center1.to(device))
    distances_1 = torch.sqrt(torch.sum((data.to(device) - center1.to(device)) ** 2, dim=2))
    distances_1_tmp = distances_1.clone()
    #print("dis_1:",distances_1)
    
    #print("cluster_num",cluster_num)
    #print("distance_1:",distances_1.shape)
    #_, t1 = torch.topk(distances_1, cluster_num, largest = False)
    #print("cluster_num",cluster_num)
    #print("distance_1:",distances_1.shape)
    #pdb.set_trace()
    '''for j in range(0, t1.shape[0]):
        for i in range(0, cluster_num):
            distances_1[j][t1[j][i]] = 0

    distances_1 = distances_1_tmp - distances_1
    '''
    distances_1 = distances_1.detach().cpu().numpy()
    # 假设distances_1和dc都是numpy数组
    # 使用np.arange和np.reshape创建一个与distances_1形状相同的索引矩阵
    index = cp.arange(distances_1.size).reshape(distances_1.shape)
    distances_1 = cp.array(distances_1)
    # 使用np.take_along_axis从distances_1中取出每行的第二个元素，作为nearest_dis
    nearest_dis = cp.take_along_axis(distances_1, index[:, 1:2], axis=1)

    # 使用np.where判断distances_1中不为零的元素，并对它们进行计算
    distances_1 = cp.where(distances_1 != 0, dc * nearest_dis / (distances_1 * distances_1), distances_1)
    '''
    for i in range(0, distances_1.shape[0]):
        nearest_dis = distances_1[i][1]
        for j in range(0, distances_1.shape[1]):
            if distances_1[i][j] != 0:
                distances_1[i][j] = dc * nearest_dis / (distances_1[i][j] * distances_1[i][j])
    '''
    distances_1 = distances_1 - 9e-15
    distances_1 = torch.Tensor(distances_1)
    #pdb.set_trace()
    distances_2 = torch.t(distances_1)
    #拼接得到最终注意力矩阵
    final_data = torch.cat((ori_data.to(device),distances_2.to(device)),dim=0)
    medium_data = torch.cat((distances_1.to(device),z_3.to(device)))
    final_data = torch.cat((final_data.to(device),medium_data.to(device)),dim=1)
    #pdb.set_trace()
    return final_data
    '''distances_2 = distances_1.T

    final_data = cp.concatenate((ori_data, distances_2), axis=0)
    medium_data = cp.concatenate((distances_1, z_3))
    final_data = cp.concatenate((final_data, medium_data), axis=1)
    return torch.Tensor(final_data)
    '''
def get_center_2(data,cluster_num):
    data =data.to(device)
    ori_data = pdist(data)
    ori_data = squareform(ori_data)
    ori_data = torch.Tensor(ori_data)
    m=5
    _,dc,_,loc = get_dc(data,m,size=2)
    ori_data_tmp = ori_data
    for j in range(0, loc.shape[0]):
        for i in range(0, cluster_num):
            ori_data[j][loc[j][i]] = 0
    ori_data = ori_data_tmp- ori_data
    for i in range(0, ori_data.shape[0]):
        for j in range(0, ori_data.shape[1]):
            if ori_data[i][j] != 0:
                ori_data[i][j] = dc * ori_data[i][1]/ori_data[i][j] * ori_data[i][j]
    ori_data -= 9e-15


    kmeans = KMeans(
        n_clusters=3, init="k-means++"
    ).fit(data.cupy())
    center = kmeans.cluster_centers_
    center1 = torch.tensor(center)
    data = data.to(device)
    data1 = torch.cat((data.to(device), center1.to(device)), dim=0)
    y_3 = pdist(center1.cupy()) #detach().cpu().numpy()
    z_3 = squareform(y_3)
    z_3 = torch.zeros(center1.shape[0],center1.shape[0]) -9e15 #邻接 and 邻接
    n = data.size(0)  # 数据1的点数
    m = center1.size(0)  # 数据2的点数


    # 计算欧几里得距离,得到n*m矩阵
    data = data.unsqueeze(1).expand(n, m, -1)  # 扩展data1的维度为(n, m, feature_dim1)
    center1 = center1.unsqueeze(0).expand(n, m, -1)  # 扩展data2的维度为(n, m, feature_dim2)
    distances_1 = torch.sqrt(torch.sum((data.to(device) - center1.to(device)) ** 2, dim=2))
    distances_1_tmp = distances_1
    _, t1 = torch.topk(distances_1,cluster_num,largest = False)
    for j in range(0, t1.shape[0]):
        for i in range(0, cluster_num):
            distances_1[j][t1[j][i]] = 0
    distances_1 = distances_1_tmp - distances_1
    for i in range(0, distances_1.shape[0]):
        for j in range(0, distances_1.shape[1]):
            if distances_1[i][j] != 0:
                distances_1[i][j] = dc * distances_1[i][1] / distances_1[i][j + 1] * distances_1[i][j + 1]
    distances_1 -= 9e-15

    '''# 计算欧几里得距离,得到m*n矩阵
    data = data.unsqueeze(1).expand(m, n, -1)  # 扩展data1的维度为(n, m, feature_dim1)
    center1 = center1.unsqueeze(0).expand(m, n, -1)  # 扩展data2的维度为(n, m, feature_dim2)
    distances_2 = torch.sqrt(torch.sum((data - center1) ** 2, dim=2))'''
    distances_2 = torch.t(distances_1)
    #拼接得到最终注意力矩阵
    final_data = torch.cat((ori_data.to(device),distances_2.to(device)),dim=0)
    medium_data = torch.cat((distances_1.to(device),z_3.to(device)))
    final_data = torch.cat((final_data.to(device),medium_data.to(device)),dim=1)
    return final_data

def get_center(data, s,size=2):
    # print(data)
    kmeans = KMeans(
        n_clusters=3, init="k-means++"
    ).fit(data)
    center = kmeans.cluster_centers_
    # print(center)
    center1 = torch.tensor(center)
    dis = np.matmul(data.cupy(), center.T)
    dis = torch.tensor(dis)
    _, t1 = torch.topk(dis, s, largest=False)
    _, t2 = torch.topk(dis, s, largest=True)
    # print(t1)
    dist1 = torch.zeros((len(data), s, size))
    for i in range(0, t1.size(0)):
        for j in range(0, t1.size(1)):
            dist1[i][j] = center1[int(t1[i][j])]
    dist2 = torch.zeros((len(data), s, size))      #dist1,dist2记录最接近和最不接近的聚类中心点
    for i in range(0, t1.size(0)):
        for j in range(0, t1.size(1)):
            dist2[i][j] = center1[int(t2[i][j])]
    # a = t1[:, 0].mean()
    return dist1,dist2,dis
    
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
        self.a = self.a.cpu()
        self.W =self.W.cpu()
    
    def forward(self, h, adj):
        h = h.to(device)
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        #print("Wh.size:",Wh.size())
        e = self._prepare_attentional_mechanism_input(Wh)
        #print("e.size:",e.size())
        knn_num = 5 
        gravity_mat = get_center_1(h,knn_num)
        
        #attention = get_center_1_1(ori_data,h,2)
        #create learnable and unlearnable contacation matrixs
        attention = e.clone()
        cluster_centers = 3
        zero_vec = -9e15*torch.ones_like(e)
        self.tensor0 = torch.zeros(cluster_centers, cluster_centers).cuda()
        self.tensor2 = nn.Parameter(torch.randn((cluster_centers, attention.shape[0]), dtype=torch.float).cuda())
        self.tensor1 = nn.Parameter(torch.randn((attention.shape[0], cluster_centers), dtype=torch.float).cuda())
        
        
        
        #must to expand the dimension of attention_matrix
        nn.init.normal_(self.tensor1,0,1)
        nn.init.normal_(self.tensor2,0,1)
        #nn.init.normal_(self.tensor3,0,1)
        #print("tensor2.device:",self.tensor2.device)
        #print("tensor2.size:",self.tensor2.size())
        attention = torch.cat((attention,self.tensor2),dim=0)
        medium_attention = torch.cat((self.tensor1,self.tensor0),dim=0)
        attention = torch.cat((attention,medium_attention),dim=1)
        #print("attention.device:",attention.device)
        #print("attention.size:",attention.size())
        #attention = torch.where(adj > 0, e, zero_vec)
        attention = torch.matmul(gravity_mat,attention) 
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        #plan1: for cluster centers fill 0 to match the dimension
        #print("h.size:",h)
        h = torch.cat((h,torch.zeros(3,h.shape[1]).cuda()),dim=0)
        '''
        tensor0 = torch.zeros(cluster_centers, cluster_centers)
        tensor0 = tensor0.cuda()
        tensor3 = nn.Parameter(torch.randn((cluster_centers, attention.shape[0]),dtype=torch.float).cuda())
        tensor4 = nn.Parameter(torch.randn((attention.shape[0], cluster_centers), dtype=torch.float).cuda())
        
        #nn.init.normal_(self.tensor0,0,1)
        nn.init.normal_(tensor3,0,1)
        nn.init.normal_(tensor4,0,1)
        print("Wh.size:",Wh.size())
        print("tensor3.size:",tensor3.size())
        print("tensor0.size:",tensor0.size())
        print("tensor4.size:",tensor4.size())
        Wh = torch.cat((Wh,tensor0),dim=0)
        medium_Wh = torch.cat((tensor4,tensor0),dim=0)
        Wh = torch.cat((Wh,medium_Wh),dim=1)
        #print("attention.size:",attention.size())
        print("Wh.size:",Wh.size())
        attention = attention.to(torch.float32)
        '''
        Wh = torch.mm(h,self.W)
        
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
        '''
        kmeans = KMeans(
            n_clusters=3, init="k-means++"
        ).fit(cp.asarray(h).get())
        center = kmeans.cluster_centers_
        # print(center)
        center1 = torch.tensor(center)
        h = torch.cat((h.to(device), center1.to(device)), dim=0)
        h = h.to(torch.float32)
        #print(h.dtype)
        #print(self.W.dtype)
        Wh = torch.mm(h.to(device), self.W)
        h_prime = torch.matmul(attention, Wh)
         
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
        '''
    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        #pdb.set_trace()
        #print("Wh.size:",Wh.size())
        #print(Wh.device)
        #print("self.a.size:",self.a[:self.out_features, :].size())
        #print(self.a.device)
        #Wh1 = torch.mm(Wh, self.a[:self.out_features, :])
        Wh2 = torch.mm(Wh, self.a[self.out_features:, :])
        #pdb.set_trace()
        #Wh2 = torch.mm(Wh, self.a[self.out_features:, :])
        Wh1 = torch.mm(Wh, self.a[:self.out_features, :])
        #pdb.set_trace()
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime
