# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
from model import ft_net
from random_erasing import RandomErasing
import json
from shutil import copyfile

# from loss import TripletLoss
version = torch.__version__
from samplers import RandomIdentitySampler
from lr_scheduler import LRScheduler
from triplet_loss import TripletLoss, CrossEntropyLabelSmooth
import cv2
import datetime
from torch.nn.functional import normalize
######################################################################
# Options
# --------

gpu_ids = '1'
name = 'review_7'
data_dir = '/home/panpan/SunJia/unsupervised_reid/code/unsupervised/examples/data/market1501/Market-1501-v15.09.15/pytorch'
train_all_1 = 'True'
batchsize = 32
erasing_p = 0.5
str_ids = gpu_ids.split(',')
gpu_ids = []
if not os.path.exists('./model/%s' % name):
    os.makedirs('./model/%s' % name)
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
# print(gpu_ids[0])


######################################################################
# Load Data
# ---------
#

transform_train_list = [
    transforms.Resize([288, 144]),
    # transforms.RandomCrop((256,128)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.Pad(10),
    transforms.RandomCrop([288, 144]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list = [
    transforms.Resize(size=(288, 144), interpolation=3),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(probability=erasing_p)]

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}

train_all = ''
if train_all_1:
    train_all = '_all'

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                               data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                             data_transforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,
                                              num_workers=8)  # 8 workers may work faster shuffle=True
               for x in ['train']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

since = time.time()
inputs, classes = next(iter(dataloaders['train']))
print(time.time() - since)

y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

def eigenvalue_power(A, max_itrs, min_delta):
    """
    mat A
    max_itrs maximum number of iterations
    min_delta Stop iteration threshold
    """
    itrs_num = 0
    delta = float('inf')
    N, _ = A.size()
    x = torch.ones(N, 1, device='cuda')
    m = 0
    #x = np.array([[0],[0],[1]])
    while itrs_num < max_itrs or delta < min_delta:
        m_1 = m
        itrs_num += 1
        y = torch.mm(A, x)
        m = y.max()
        x = torch.div(y, m)
        delta = abs(m - m_1)
    return m

def dominant_eigenvalue(A: 'N x N'):
    N, _ = A.size()
    x = torch.randn(N, 1, device='cuda')
    for i in range(2):
        x = torch.mm(A, x)
        numerator = torch.mm(A, x).squeeze()
        denominator = (torch.mm(A, torch.mm(A, x))).squeeze()
    numerator = (torch.norm(numerator, p=2) ** 2).squeeze()
    denominator = (torch.norm(denominator, p=2) ** 2).squeeze()
    return numerator / (denominator + 0.1)

def get_eigenvalue(A: 'M x N, M >= N'):
    #ATA = A.permute(1, 0) @ A
    N, _ = A.size()
    largest = dominant_eigenvalue(A)
    #largest = eigenvalue_power(A, 20, 1e-4)
    I = torch.eye(N, device='cuda')  # noqa
    I = I * largest  # noqa
    tmp = dominant_eigenvalue(A - I)
    #tmp = eigenvalue_power(A-I, 20, 1e-4)
    return tmp + largest, largest

def l2_reg_ortho(mdl):
    with torch.no_grad():
        l2_reg = None
        for W in mdl.parameters():
            if W.ndimension() < 2:
                continue
            else:
                cols = W[0].numel()
                w1 = W.view(-1, cols)
                wt = torch.transpose(w1, 0, 1)
                m = torch.matmul(wt, w1)
                smallest, largest = get_eigenvalue(m)

                # ident = Variable(torch.eye(cols,cols))
                # ident = ident.cuda()
                #
                # w_tmp = (m - ident)
                # height = w_tmp.size(0)
                # u = normalize(w_tmp.new_empty(height).normal(0,1), dim=0, eps=1e-12)
                # v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
                # u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
                # sigma = torch.dot(u, torch.matmul(w_tmp, v))
                sigma = ((largest - 1) * (smallest - 1)) ** 2
                if l2_reg is None:
                    l2_reg = sigma
                else:
                    l2_reg = l2_reg + sigma
        return l2_reg


def adjust_ortho_decay_rate(epoch):
    o_d = 1e-3
    if epoch > 160:
        o_d = 0.0
    elif epoch > 120:
        o_d = 1e-3 * o_d
    elif epoch > 70:
        o_d = 1e-2 * o_d
    elif epoch > 30:
        o_d = 1e-1 * o_d
    return o_d

def train_base_model(model, criterion, triplet, num_epochs):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # update learning rate
        lr_scheduler = LRScheduler(base_lr=3e-2, step=[60, 130],
                                   factor=0.1, warmup_epoch=10, warmup_begin_lr=3e-4)

        lr = lr_scheduler.update(epoch)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
        print(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                # scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs

                inputs, labels = data
                now_batch_size, c, h, w = inputs.shape
                if now_batch_size < batchsize:  # skip the last batch
                    continue
                # print(inputs.shape)
                # wrap them in Variable
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                temp_loss = []
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs1, outputs2, outputs3, outputs4= model(inputs)

                odecay = adjust_ortho_decay_rate(epoch + 1)
                oloss = l2_reg_ortho(model)
                oloss = odecay * oloss
                # _, preds = torch.max(outputs.data, 1)
                _, preds1 = torch.max(outputs1.data, 1)
                _, preds2 = torch.max(outputs2.data, 1)
                _, preds3 = torch.max(outputs3.data, 1)
                _, preds4 = torch.max(outputs4.data, 1)

                loss1 = criterion(outputs1, labels)
                loss2 = criterion(outputs2, labels)
                loss3 = criterion(outputs3, labels)
                loss4 = criterion(outputs4, labels)

                temp_loss.append(loss1)
                temp_loss.append(loss2)
                temp_loss.append(loss3)
                temp_loss.append(loss4)

                loss = sum(temp_loss) / 4 + oloss

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * now_batch_size
                a = float(torch.sum(preds1 == labels.data))
                b = float(torch.sum(preds2 == labels.data))
                c = float(torch.sum(preds3 == labels.data))
                d = float(torch.sum(preds4 == labels.data))

                running_corrects_1 = a + b + c + d
                running_corrects_2 = running_corrects_1 / 4
                running_corrects += running_corrects_2
                # running_corrects +=float(torch.sum(preds == labels.data))
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            # 在日志文件中记录每个epoch的精度和loss
            with open('./model/%s/%s.txt' % (name, name), 'a') as acc_file:
                acc_file.write(
                    'lr: %.8f, Epoch: %2d, Precision: %.8f, Loss: %.8f\n' % (lr, epoch, epoch_acc, epoch_loss))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)
            # deep copy the model
            if phase == 'train':
                last_model_wts = model.state_dict()
                if epoch < 150:
                    if epoch % 10 == 9:
                        save_network(model, epoch)
                    draw_curve(epoch)
                else:
                    # if epoch%2 == 0:
                    save_network(model, epoch)
                    draw_curve(epoch)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    return model


######################################################################
# Draw Curve
# ---------------------------

x_epoch = []
fig = plt.figure(figsize=(32, 16))
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")

def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    # ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    # ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join('./model', name, 'train.jpg'))


def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', name, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])


def main():
    time_start = time.time()
    model = ft_net(len(class_names))
    if use_gpu:
        model = model.cuda()
    train_base_model(model, criterion, triplet, num_epochs=220)

    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Elapsed {}'.format(elapsed))

if __name__ == '__main__':
    triplet = TripletLoss(margin=0.3)
    criterion = CrossEntropyLabelSmooth(num_classes=len(class_names))
    dir_name = os.path.join('./model', name)
    if os.path.isdir(dir_name):
        # os.mkdir(dir_name)
        copyfile('./train_11.py', dir_name + '/train_11.py')
        copyfile('./model.py', dir_name + '/model.py')
    main()
