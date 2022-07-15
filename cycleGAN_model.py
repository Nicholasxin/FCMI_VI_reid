import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.nn import init

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ResidualBlock(nn.Module):
    '''Residual Block with Instance Normalization'''

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
        )

    def forward(self, x):
        return self.model(x) + x


class Generator(nn.Module):
    '''Generator with Down sampling, Several ResBlocks and Up sampling.
       Down/Up Samplings are used for less computation.
    '''

    def __init__(self, conv_dim, layer_num):
        super(Generator, self).__init__()

        layers = []

        # input layer
        layers.append(nn.Conv2d(in_channels=3, out_channels=conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # down sampling layers
        current_dims = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(current_dims, current_dims*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(current_dims*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            current_dims *= 2

        # Residual Layers
        for i in range(layer_num):
            layers.append(ResidualBlock(current_dims, current_dims))

        # up sampling layers
        for i in range(2):
            layers.append(nn.ConvTranspose2d(current_dims, current_dims//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(current_dims//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            current_dims = current_dims//2

        # output layer
        layers.append(nn.Conv2d(current_dims, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

    def forward(self, x,):
        return self.model(x)


class Discriminator(nn.Module):
    '''Discriminator with PatchGAN'''


    def __init__(self, conv_dim, layer_num):
        super(Discriminator, self).__init__()

        current_dim = conv_dim
        feature_convs = []
        feature_convs.append(nn.Conv2d(2048, current_dim, kernel_size=1, stride=1, padding=0))
        feature_convs.append(nn.LeakyReLU(0.2, inplace=True))
        self.feature_convs = nn.Sequential(*feature_convs)

        dis_convs = []
        dis_convs.append(nn.Conv2d(current_dim, current_dim*2, kernel_size=4, stride=2, padding=1))
        dis_convs.append(nn.InstanceNorm2d(current_dim*2))
        dis_convs.append(nn.LeakyReLU(0.2, inplace=True))
        dis_convs.append(nn.Conv2d(current_dim*2, current_dim*2, kernel_size=4, stride=2, padding=1))
        dis_convs.append(nn.InstanceNorm2d(current_dim * 2))
        dis_convs.append(nn.Sigmoid())
        self.dis_convs = nn.Sequential(*dis_convs)

    def forward(self, features):
        features = self.feature_convs(features)
        #features = F.interpolate(features, [16, 16], mode='bilinear')
        out_src = self.dis_convs(features)
        return out_src


class ConditionalDiscriminator(nn.Module):
    '''Discriminator with PatchGAN'''

    def __init__(self, conv_dim):
        super(ConditionalDiscriminator, self).__init__()

        self.sigmoid = nn.Sigmoid()

        # image convs
        image_convs = []
        image_convs.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        image_convs.append(nn.LeakyReLU(0.2, inplace=True))
        current_dim = conv_dim
        for i in range(3):
            image_convs.append(nn.Conv2d(current_dim, current_dim, kernel_size=4, stride=2, padding=1))
            image_convs.append(nn.InstanceNorm2d(current_dim))
            image_convs.append(nn.LeakyReLU(0.2, inplace=True))
            #current_dim *= 2
        self.image_convs = nn.Sequential(*image_convs)

        # feature convs
        feature_convs = []
        feature_convs.append(nn.Conv2d(2048, conv_dim, kernel_size=1, stride=1, padding=0))
        feature_convs.append(nn.LeakyReLU(0.2, inplace=True))
        self.feature_convs = nn.Sequential(*feature_convs)

        # discriminator convs
        dis_convs = []
        dis_convs.append(nn.Conv2d(current_dim+conv_dim, current_dim*2, kernel_size=4, stride=2, padding=1))
        dis_convs.append(nn.InstanceNorm2d(current_dim*2))
        dis_convs.append(nn.LeakyReLU(0.2, inplace=True))
        current_dim *= 2
        dis_convs.append(nn.Conv2d(current_dim, current_dim * 2, kernel_size=4, stride=2, padding=1))
        dis_convs.append(nn.InstanceNorm2d(current_dim * 2))
        dis_convs.append(nn.LeakyReLU(0.2, inplace=True))
        current_dim *= 2
        self.dis_convs = nn.Sequential(*dis_convs)

        # output layer
        self.conv_src = nn.Conv2d(current_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, images, features):
        images = self.image_convs(images)
        features = self.feature_convs(features)
        #features = F.interpolate(features, [16, 16], mode='bilinear')
        x = torch.cat([images, features], dim=1)
        x = self.dis_convs(x)
        out_src = self.conv_src(x)
        out_src = self.sigmoid(out_src)
        return out_src
