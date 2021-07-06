import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import arguments as args
#args.label_dim args.data_dim args.noise_dim#
'''Discriminator'''
def weights_init(m):

    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Linear') != -1:

        nn.init.constant_(m.weight.data, 1.0)
        # if m.weight.data.shape
        if m.weight.data.shape[0] == 2:
            m.weight.data[0,1], m.weight.data[1,0] = 0, 0
        # print(m.weight.data)
        # m.weight.data.normal_(0.0, 1.0)
        nn.init.constant_(m.bias.data, 0)
        # nn.init.constant_(m.bias.data, 0)
        # m.bias.data.constant_(0)
        # nn.init.constant_(m.bias.data, 0)
    #
    # elif classname.find('BatchNorm') != -1:
    #     m.weight.data.normal_(1.0, 0.02)






class Discriminator(nn.Module):
    def __init__(self, class_size):
        super(Discriminator, self).__init__()

        self.embedding = nn.Embedding(class_size, args.label_dim)


        self.seq = nn.Sequential(
            nn.Linear(args.data_dim + args.label_dim, 40, bias=True),
            # nn.BatchNorm1d(40),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Linear(512, 128),
            # nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(40, 1),
            # nn.Sigmoid()
        )
        # self.apply(weights_init)
        # input()

    def forward(self, input, label):
        #embed label
        #input [batch_size, pca_dim]
        label = self.embedding(label)

        #flatten image to 2d tensor
        # input = input.view(input.size(0), -1)

        #concatenate image vector and label
        label = torch.squeeze(label, 1)
        # print(input.shape, label.shape)
        x = torch.cat([input, label], 1)
        result = self.seq(x)#[batch_size, 1]

        return result







'''Generator'''
class Generator(nn.Module):
    def __init__(self, vt, latent_size, class_size):
        super(Generator, self).__init__()

        # for p in self.parameters():
        #     p.requires_grad = False
            # print(p)
        # input()
        self.vt = vt
        self.embedding = nn.Embedding(class_size, args.label_dim)
        self.seq = nn.Sequential(
            nn.Linear(args.noise_dim + args.label_dim, 500, bias=True),
            # nn.BatchNorm1d(40),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(500, args.data_dim),
            nn.Tanh())
        if not self.vt:
            for p in self.parameters():
                # print(p)
                p.requires_grad = False
                # print(p)
            self.noise = nn.Sequential(nn.Linear(args.noise_dim, args.noise_dim, bias=True))
            self.apply(weights_init)
        # input()

    def forward(self, input, label):
        #embed label
        #input [batch_size, noise_dim]
        #label [batch_size]
        # print(label.shape)
        if not self.vt:
            input = self.noise(input)
            # input()
        label = self.embedding(label)
        label = torch.squeeze(label, 1)
        # print(label.shape)
        #concatenate latent vector (input) and label
        # print(input.shape, label.shape)
        x = torch.cat([input, label], 1)#[batch_size, noise_dim+num_labels]
        result = self.seq(x)#[batch_size, data_dim]
        # result = result.view(-1, 1, 28, 28)

        return result
