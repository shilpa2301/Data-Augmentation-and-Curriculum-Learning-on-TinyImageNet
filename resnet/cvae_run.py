import random
import shutil
from hashlib import md5
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
import cv2
from tqdm import tqdm
import torch
import numpy as np
import time
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torchvision
import os, time, tqdm
from torchvision import datasets, transforms
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(42)

kwargs = {'num_workers': 4, 'pin_memory': True}

# hyper params
batch_size = 64
latent_size = 20
epochs = 10

class EarlyStop:
    """Used to early stop the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, verbose=False, delta=0,
                 save_name="checkpoint.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            save_name (string): The filename with which the model and the optimizer is saved when improved.
                            Default: "checkpoint.pt"
        """
        self.patience = patience
        self.verbose = verbose
        self.save_name = save_name
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, optimizer):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, model, optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        state = {"net":model.state_dict(), "optimizer":optimizer.state_dict()}
        torch.save(state, self.save_name)
        self.val_loss_min = val_loss

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class MLP(nn.Module):
    def __init__(self, hidden_size, last_activation = True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(hidden_size)-1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i+1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_size)-2) or ((i == len(hidden_size) - 2) and (last_activation)):
                q.append(("BatchNorm_%d" % i, nn.BatchNorm1d(out_dim)))
                q.append(("ReLU_%d" % i, nn.ReLU(inplace=True)))
        self.mlp = nn.Sequential(OrderedDict(q))
    def forward(self, x):
        return self.mlp(x)

class Encoder(nn.Module):
    def __init__(self, shape, nhid = 16, ncond = 0):
        super(Encoder, self).__init__()
        c, h, w = shape
        ww = ((w-8)//2 - 4)//2
        hh = ((h-8)//2 - 4)//2
        self.encode = nn.Sequential(nn.Conv2d(c, 16, 5, padding = 0), nn.BatchNorm2d(16), nn.ReLU(inplace = True),
                                    nn.Conv2d(16, 32, 5, padding = 0), nn.BatchNorm2d(32), nn.ReLU(inplace = True),
                                    nn.MaxPool2d(2, 2),
                                    nn.Conv2d(32, 64, 3, padding = 0), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
                                    nn.Conv2d(64, 64, 3, padding = 0), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
                                    nn.MaxPool2d(2, 2),
                                    Flatten(), MLP([ww*hh*64, 256, 128])
                                   )
        self.calc_mean = MLP([128+ncond, 64, nhid], last_activation = False)
        self.calc_logvar = MLP([128+ncond, 64, nhid], last_activation = False)
    def forward(self, x, y = None):
        x = self.encode(x)
        if (y is None):
            return self.calc_mean(x), self.calc_logvar(x)
        else:
            return self.calc_mean(torch.cat((x, y), dim=1)), self.calc_logvar(torch.cat((x, y), dim=1))

class Decoder(nn.Module):
    def __init__(self, shape, nhid = 16, ncond = 0):
        super(Decoder, self).__init__()
        c, w, h = shape
        self.shape = shape
        self.decode = nn.Sequential(MLP([nhid+ncond, 64, 128, 256, c*w*h], last_activation = False), nn.Sigmoid())
    def forward(self, z, y = None):
        c, w, h = self.shape
        if (y is None):
            return self.decode(z).view(-1, c, w, h)
        else:
            return self.decode(torch.cat((z, y), dim=1)).view(-1, c, w, h)

class VAE(nn.Module):
    def __init__(self, shape, nhid = 16):
        super(VAE, self).__init__()
        self.dim = nhid
        self.encoder = Encoder(shape, nhid)
        self.decoder = Decoder(shape, nhid)

    def sampling(self, mean, logvar):
        eps = torch.randn(mean.shape).to(device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.sampling(mean, logvar)
        return self.decoder(z), mean, logvar

    def generate(self, batch_size = None):
        z = torch.randn((batch_size, self.dim)).to(device) if batch_size else torch.randn((1, self.dim)).to(device)
        res = self.decoder(z)
        if not batch_size:
            res = res.squeeze(0)
        return res

class cVAE(nn.Module):
    def __init__(self, shape, nclass, nhid = 16, ncond = 16):
        super(cVAE, self).__init__()
        self.dim = nhid
        self.encoder = Encoder(shape, nhid, ncond = ncond)
        self.decoder = Decoder(shape, nhid, ncond = ncond)
        self.label_embedding = nn.Embedding(nclass, ncond)

    def sampling(self, mean, logvar):
        eps = torch.randn(mean.shape).to(device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, x, y):
        y = self.label_embedding(y)
        mean, logvar = self.encoder(x, y)
        z = self.sampling(mean, logvar)
        return self.decoder(z, y), mean, logvar

    def generate(self, class_idx):
        if (type(class_idx) is int):
            class_idx = torch.tensor(class_idx)
        class_idx = class_idx.to(device)
        if (len(class_idx.shape) == 0):
            batch_size = None
            class_idx = class_idx.unsqueeze(0)
            z = torch.randn((1, self.dim)).to(device)
        else:
            batch_size = class_idx.shape[0]
            z = torch.randn((batch_size, self.dim)).to(device)
        y = self.label_embedding(class_idx)
        res = self.decoder(z, y)
        if not batch_size:
            res = res.squeeze(0)
        return res

BCE_loss = nn.BCELoss(reduction = "sum")
bce_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
# def MSE_loss (X_hat, X):
#     return ((X_hat-X)**2)/len(X_hat)



def loss(X, X_hat, mean, logvar):
    # print(f"X={X.shape}")
    reconstruction_loss = bce_loss_fn(X_hat, X)
    
    # reconstruction_loss = BCE_loss(X_hat, X)
    # reconstruction_loss = MSE_loss(X_hat, X)

    KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean**2)
    return reconstruction_loss + KL_divergence

if __name__ == "__main__":

    train_transform = transforms.Compose([
    # transforms.Resize(32),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262]),
    ])

    test_transform = transforms.Compose([
        # transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262]),
    ])

    # train_transform = transforms.Compose([
    # # transforms.Resize(32),
    # transforms.RandomRotation(20),
    # transforms.RandomHorizontalFlip(0.5),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # ])

    # test_transform = transforms.Compose([
    #     # transforms.Resize(32),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),

    # ])

    train_dataset = datasets.ImageFolder(root='/home/csgrad/dl_228/tiny-imagenet-200/train', transform=train_transform)
    val_dataset = datasets.ImageFolder(root='/home/csgrad/dl_228/tiny-imagenet-200/val', transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False, **kwargs)

    print(len(train_loader))


    net = cVAE((3, 64, 64), 200, nhid = 2, ncond = 16)
    net.to(device)
    print(net)
    save_name = "cVAE.pt"

    ############### training #########################
    train_iter = train_loader
    lr = 0.01
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay = 0.0001)

    def adjust_lr(optimizer, decay_rate=0.95):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_rate

    retrain = True
    if os.path.exists(save_name):
        print("Model parameters have already been trained before. Retrain ? (y/n)")
        ans = input()
        if not (ans == 'y'):
            checkpoint = torch.load(save_name, map_location = device)
            net.load_state_dict(checkpoint["net"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            for g in optimizer.param_groups:
                g['lr'] = lr

    max_epochs = 1000
    early_stop = EarlyStop(patience = 20, save_name = save_name)
    net = net.to(device)

    print("training on ", device)
    for epoch in range(max_epochs):

        train_loss, n, start = 0.0, 0, time.time()
        for X, y in tqdm.tqdm(train_iter, ncols = 50):
            X = X.to(device)
            y = y.to(device)
            X_hat, mean, logvar = net(X, y)

            # print(f"X_hat, mean, logvar={X_hat.shape}, {mean.shape}, {logvar.shape}")
            l = loss(X, X_hat, mean, logvar).to(device)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_loss += l.cpu().item()
            n += X.shape[0]

        train_loss /= n
        print('epoch %d, train loss %.4f , time %.1f sec'
              % (epoch, train_loss, time.time() - start))

        adjust_lr(optimizer)

        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"{save_name}_epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': train_loss
            }, checkpoint_path)
            print(f"Model saved at {checkpoint_path}")

        if (early_stop(train_loss, net, optimizer)):
            break

    checkpoint = torch.load(early_stop.save_name)
    net.load_state_dict(checkpoint["net"])