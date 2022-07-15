import numpy as np
from torch import nn, tensor
import torch
from torch.autograd import Variable
import torch.nn.functional as F
def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x
def cosine(inputs_1, inputs_2):
    '''
    :param inputs_1: torch.tensor, 2d
    :param inputs_2: torch.tensor, 2d
    :param labels: 2d,
    :return:
    '''
    bs1 = normalize(inputs_1,axis=1)
    bs2 = normalize(inputs_2,axis=1)
    dist = -torch.mm(bs1, bs2.t())
    dist+=1
    return dist

class hetero_loss(nn.Module):
    def __init__(self, margin=0.1):
        super(hetero_loss, self).__init__()
        self.margin = margin
        self.dist = nn.MSELoss(reduction='sum')

    def forward(self, feat11, feat22, label1, label2):
        feat1 = feat11
        feat2 = feat22
        feat_size = feat1.size()
        feat_num = feat1.size()
        label_num = len(label1.unique())
        feat1 = feat1.chunk(label_num, 0)
        feat2 = feat2.chunk(label_num, 0)
        # loss = Variable(.cuda())
        for i in range(label_num):
            center1 = torch.mean(feat1[i], dim=0)
            center2 = torch.mean(feat2[i], dim=0)
            if i == 0:
                dist = max(0, self.dist(center1, center2) - self.margin)
            else:
                dist += max(0, self.dist(center1, center2) - self.margin)

        return dist

class smsc_loss (nn.Module):
    #same modality same calss#

    def __init__(self, margin=0.1):
        super(smsc_loss, self).__init__()
        self.margin = margin
        #self.dist = nn.MSELoss(reduction='sum')
        self.dist = nn.CosineSimilarity(dim=0)

    def forward(self, feat, label):
        feat_size = feat.size()[1]
        feat_num = feat.size()[0]
        label_num = len(label.unique())
        feat_single = feat.chunk(feat_num, 0)
        feat_center = feat.chunk(label_num, 0)
        for i in range(label_num):
            center = torch.mean(feat_center[i], dim=0)
            for j in range(feat_center[i].size()[0]):
                feat_single1 = torch.squeeze(feat_single[j])
                if j == 0:
                    dist = max(0, 1 - (self.dist(center, feat_single1)))
                else:
                    dist += max(0, 1 - (self.dist(center, feat_single1)))

        return dist


class smdc_loss(nn.Module):
    # same modality different calss#

    def __init__(self, margin=0.1):
        super(smdc_loss, self).__init__()
        self.margin = margin
        self.dist = nn.CosineSimilarity(dim=0)

    def forward(self, feat, label):
        feat_size = feat.size()[1]
        feat_num = feat.size()[0]
        n = len(label.unique())
        feat = feat.chunk(n, 0)
        dist_list = []
        center_list1 = []
        for i in range(n):
            center = torch.mean(feat[i], dim=0)
            center_list1.append(center)
        for i in range(n):
            for j in range(n):
                dist1 =self.dist(center_list1[i], center_list1[j])
                dist_list.append(dist1)
        dist = torch.tensor(dist_list, requires_grad = True).view(n,n).cuda()
        label1 = label.unique()
        mask = label1.expand(n, n).eq(label1.expand(n, n).t())
        dist_an = []
        for i in range(n):
            dist_an.append(dist[i][mask[i] == 0].min().view(1))
        for i in range(n):
            if i == 0:
                dist =1-dist_an[i]
            else:
                dist += 1-dist_an[i]
        dist = torch.squeeze(dist)
        return dist

# class dmsc_loss(nn.Module):
#     # different modality same calss#
#
#     def __init__(self, margin=0.1):
#         super(dmsc_loss, self).__init__()
#         self.margin = margin
#         #self.dist = nn.CosineSimilarity(dim=0)
#
#     def forward(self, feat11, feat22, label1,label2):
#         feat1 = feat11[0]
#         feat2 = feat22[0]
#         feat_size = feat1.size()[1]
#         feat_num = feat1.size()[0]
#         label_num = len(label1.unique())
#         feat1 = feat1.chunk(label_num, 0)
#         feat2 = feat2.chunk(label_num, 0)
#         # loss = Variable(.cuda())
#         for i in range(label_num):
#             center1 = torch.mean(feat1[i], dim=0)
#             center2 = torch.mean(feat2[i], dim=0)
#             if i == 0:
#                 dist = max(0, cosine(center1, center2) - self.margin)
#             else:
#                 dist += max(0, cosine(center1, center2) - self.margin)
#         return dist


class dmdc_loss(nn.Module):
    # different modality different calss & different modality same calss#
    def __init__(self, margin=0.1):
        super(dmdc_loss, self).__init__()
        self.margin = margin
        self.dist = nn.CosineSimilarity(dim=0)

    def forward(self, feat1, feat2, label1, label2):
        feat_size = feat1.size()[1]
        feat_num = feat1.size()[0]
        label_num = len(label1.unique())
        feat1 = feat1.chunk(label_num, 0)
        feat2 = feat2.chunk(label_num, 0)
        center11 = []
        center22 = []
        for i in range(label_num):
            center1 = torch.mean(feat1[i], dim=0)
            center2 = torch.mean(feat2[i], dim=0)
            center11.append(center1)
            center22.append(center2)
        center11 = torch.cat((center11),0).view(label_num,feat_size)
        center22 = torch.cat((center22), 0).view(label_num,feat_size)
        dist_list1 = []
        dist_list2 = []
        for i in range(label_num):
            for j in range(label_num):
                dist1 = 1 - self.dist(center11[i], center22[j])
                dist2 = 1 - self.dist(center11[i], center11[j])
                dist_list1.append(dist1)
                dist_list2.append(dist2)
        dist11 = torch.tensor(dist_list1, requires_grad = True).view(label_num, label_num).cuda()
        dist22 = torch.tensor(dist_list2, requires_grad = True).view(label_num, label_num).cuda()
        dist_final = abs(dist11 - dist22)
        dist_final1 = dist_final.sum()
        return dist_final1

class JS_div(nn.Module):
    # different modality different calss & different modality same calss#
    def __init__(self, margin=0.1):
        super(JS_div, self).__init__()
        self.margin = margin
        self.dist = nn.CosineSimilarity(dim=0)
        self.KLDivloss = nn.KLDivLoss(reduction='batchmean')
    def forward(self, feat1, feat2, get_softmax = True):
        if get_softmax:
            feat11 = F.softmax(feat1)
            feat22 = F.softmax(feat2)
        log_mean_output = ((feat11 + feat22)/2).log()
        dis_final = (self.KLDivloss(log_mean_output, feat11) + self.KLDivloss(log_mean_output, feat22))/2
        return dis_final

