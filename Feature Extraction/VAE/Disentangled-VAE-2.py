#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:50:01 2020

@author: Nick Tacca
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn

# Set initial variables
ZDIMS = 20
BETA = 10
LR = 1e-3
EPOCHS = 1
BATCH_SIZE = 10
SEED = 4
LOG_INTERVAL = 100

# VAE Initial Filters
CHAN = 1
ENCODE_F = 128
DECODE_F = 128

class EEGdata(Dataset):
    """Loading EEG dataset"""

    def __init__(self, mat_file, transform=None):
        """
            mat_file (string): Path to the mat file with feature data and labels
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.data = loadmat(os.path.join(os.getcwd(), mat_file))
        self.feats = self.data["feats"]
        self.labels = self.data["labels"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.feats[:, :, idx]
        even_img = np.zeros((sample.shape[0], sample.shape[0]-sample.shape[1]), dtype=np.float32)
        sample = np.append(sample, even_img, axis=1)
        label = self.labels[idx]

        return sample, label

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis for torch files
        feat = sample.transpose((2, 0, 1))
        feat_tensor = torch.from_numpy(feat)
        return feat_tensor

class VAE(nn.Module):
    """Beta-VAE to extract disentangled latent features"""

    def __init__(self, zdims):
        super(VAE, self).__init__()
        
        self.zdims = zdims
        self.encoder = nn.Sequential(
            
            # input is CHAN x 64 x 64 (padded end of feature-vector with zeroes)
            nn.Conv2d(CHAN, ENCODE_F, 4, 3, 1),
            nn.ReLU(inplace=True),

            # conv layer 2
            nn.Conv2d(ENCODE_F, ENCODE_F * 2, 4, 2, 1),
            nn.BatchNorm2d(ENCODE_F * 2),
            nn.ReLU(inplace=True),

            # conv layer 3
            nn.Conv2d(ENCODE_F * 2, ENCODE_F * 3, 4, 2, 1),
            nn.BatchNorm2d(ENCODE_F * 3),
            nn.ReLU(inplace=True),

            # conv layer 4
            nn.Conv2d(ENCODE_F * 3, 512, 4, 2, 0),
            nn.ReLU(inplace=True)

        )
        
        self.decoder = nn.Sequential(

            # input is Z (post-fc)
            nn.ConvTranspose2d(512, DECODE_F * 4, 4, 1, 0),
            nn.BatchNorm2d(DECODE_F * 4),
            nn.ReLU(inplace=True),

            # deconv layer 2
            nn.ConvTranspose2d(DECODE_F * 4, DECODE_F * 3, 3, 2, 1),
            nn.BatchNorm2d(DECODE_F * 3),
            nn.ReLU(inplace=True),

            # deconv layer 3
            nn.ConvTranspose2d(DECODE_F * 3, DECODE_F * 2, 4, 3, 2),
            nn.BatchNorm2d(DECODE_F * 2),
            nn.ReLU(inplace=True),

            # deconv layer 4
            nn.ConvTranspose2d(DECODE_F * 2, CHAN, 4, 4, 4),
            nn.Sigmoid()

        )
        
        # conv fc
        self.fc11 = nn.Linear(512, self.zdims) # mu
        self.fc12 = nn.Linear(512, self.zdims) # logvar
        
        # deconv fc
        self.fc2 = nn.Linear(self.zdims, 512)
    
    def loss_function(self, x_conn, x, mu, logvar, beta=BETA):
        # print(x_conn.shape)
        # print(x.shape)
        cross_entropy_loss = F.binary_cross_entropy(x_conn, x.view(BATCH_SIZE, 1, 64, 64), reduction='sum')
        KL_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        loss = cross_entropy_loss + (beta * KL_loss)
        return loss
        
    def encode(self, x):
        conv = self.encoder(x)
        # print(conv.shape)
        conv = conv.view(-1, 512)
        mu = self.fc11(conv)
        logvar = self.fc12(conv)
        return mu, logvar
    
    def decode(self, z):
        deconv_input = F.relu(self.fc2(z))
        deconv_input = deconv_input.view(-1, 512, 1, 1)
        x_conn = self.decoder(deconv_input)
        # print("x_conn: ".format(x_conn.shape))
        return x_conn
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar * 0.5)
        eps = torch.rand_like(std)
        z = eps.mul(std).add(mu)
        return z
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_conn = self.decode(z)
        return x_conn, mu, logvar, z
    
def train(vae, device, epoch, beta=BETA):

    train_loader = load_data("TRAIN")
    optimizer = optim.Adam(vae.parameters(), lr=LR)

    vae.train()
    train_loss = 0
    
    for batch, (data, _) in enumerate(train_loader):
        data = data.unsqueeze(dim=1) # grayscale images
        x = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, Z = vae(x)
        # print(recon_batch.shape)
        loss = vae.loss_function(recon_batch, x, mu, logvar, beta)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                epoch,
                batch * len(x),
                len(train_loader.dataset),
                100. * batch / len(train_loader),
                loss.item() / len(data)
            ))
    
    print('--- Epoch: {} Average loss: {:.4f}'.format(
        epoch,
        train_loss / len(train_loader.dataset)
    ))

def test(vae, device, epoch):
    test_loader = load_data("TEST")
    vae.eval()
    test_loss = 0
    
    with torch.no_grad():
        
        for i, (data, _) in enumerate(test_loader):
            data = data.unsqueeze(dim=1) # grayscale images
            x = data.to(device)
            recon_batch, mu, logvar, Z = vae(x)
            test_loss += vae.loss_function(recon_batch, x, mu, logvar).item()
            
            if i == 0:
                n = min(x.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(BATCH_SIZE, 1, 64, 64)[:n]])
                save_image(comparison.cpu(), 'results/reconstruction_' + str(epoch) + '.png', nrow = n)
    
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def device_settings(): 
    torch.manual_seed(SEED)
    if torch.cuda.is_available(): 
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    return device

def load_data(dataset):

    os.chdir('/home/nick/Documents/Hiwi/VAE')

    if dataset == "TRAIN":
        # Load training data
        train_dataset = EEGdata(mat_file="Datasets/Training_Data.mat", transform=transforms.ToTensor())
        dataset_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    elif dataset == "TEST":
        # Load test data
        test_dataset = EEGdata(mat_file="Datasets/Test_Data.mat", transform=transforms.ToTensor())
        dataset_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

    return dataset_loader
    
def main():
    # device = device_settings()
    device = torch.device("cpu")
    vae = VAE(zdims=ZDIMS).to(device)

    # Training
    for epoch in range(1, EPOCHS + 1):
        train(vae, device, epoch, beta=BETA)
        test(vae, device, epoch)
    
if __name__ == "__main__":
    main()
