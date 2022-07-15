from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model import embed_net
from utils import *
from loss import OriTripletLoss, TripletLoss_WRT
from tensorboardX import SummaryWriter
from cycleGAN_model import Discriminator,weights_init_normal,ConditionalDiscriminator
import itertools
from design_loss import *
import matplotlib.pyplot as plt
import scipy.io


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def normalize1(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x
mse_loss = nn.MSELoss().cuda()
def sim_loss(feat_rgb1, feat_ir1,feat1,L):

    rgb_cen = []
    ir_cen = []
    feat2 = []
    for i in range(L):
        feat2.append(feat1[i].mean(dim=0))
        rgb_cen.append(feat_rgb1[i].mean(dim=0))
        ir_cen.append(feat_ir1[i].mean(dim=0))
    rgb_cen = torch.stack(rgb_cen)
    ir_cen = torch.stack(ir_cen)
    feat2 = torch.stack(feat2)
    rgb_cen = normalize1(rgb_cen, axis=1)
    ir_cen = normalize1(ir_cen, axis=1)
    feat2 = normalize1(feat2, axis=1)
    rgb_martix = euclidean_dist(rgb_cen, feat2).cuda()
    ir_martix = euclidean_dist(ir_cen, feat2).cuda()
    rgb_mat_diag = torch.diag(rgb_martix).cuda()
    ir_mat_diag = torch.diag(ir_martix).cuda()

    loss_sim = 0.5*(rgb_mat_diag + ir_mat_diag).mean()

    return loss_sim


    # rgb_cen = normalize1(rgb_cen, axis=1)
    # ir_cen = normalize1(ir_cen, axis=1)
    # cen_martix = euclidean_dist(rgb_cen, ir_cen).cuda()
    # dist_mat_diag = torch.diag(cen_martix).cuda()
    # a_diag = torch.diag_embed(dist_mat_diag).cuda()
    # dif = cen_martix + 5 * a_diag.cuda()
    # pred1, idx1 = dif.min(0)
    # pred1 = torch.diag_embed(pred1).cuda()
    # loss_sim = (a_diag - pred1 + 0.3).pow(2).sqrt().mean().cuda()