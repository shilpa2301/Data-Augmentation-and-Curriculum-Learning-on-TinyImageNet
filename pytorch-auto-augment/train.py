import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils import *
from wide_resnet import WideResNet
from auto_augment import AutoAugment, Cutout

import torchvision
import pickle
device='cuda' if torch.cuda.is_available() else 'cpu'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--dataset', default='cifar10',
                        choices=['cifar10', 'cifar100', 'tinyimagenet'],
                        help='dataset name')
    parser.add_argument('--depth', default=28, type=int)
    parser.add_argument('--width', default=10, type=int)
    parser.add_argument('--cutout', default=False, type=str2bool)
    parser.add_argument('--auto-augment', default=False, type=str2bool)
    parser.add_argument('--epochs', default=270, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float)
    parser.add_argument('--milestones', default='60,120,160', type=str)
    parser.add_argument('--gamma', default=0.2, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--nesterov', default=False, type=str2bool)

    args = parser.parse_args()

    return args

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def train(args, train_loader, model, criterion, optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    scores = AverageMeter()

    model.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # from original paper's appendix
        input = input.to(device)#.cuda()
        target = target.to(device) #cuda()

        output = model(input)
        loss = criterion(output, target)

        acc = accuracy(output, target)[0]

        losses.update(loss.item(), input.size(0))
        scores.update(acc.item(), input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('acc', scores.avg),
    ])

    return log


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.to(device)#.cuda()
            target = target.to(device)#.cuda()

            output = model(input)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            scores.update(acc1.item(), input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('acc', scores.avg),
    ])

    return log


def main():
    args = parse_args()
    # device='cuda' if torch.cuda.is_available() else 'cpu'

    if args.name is None:
        args.name = '%s_VGG16_Adam' %(args.dataset)
        if args.cutout:
            args.name += '_wCutout'
        if args.auto_augment:
            args.name += '_wAutoAugment'

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    criterion = nn.CrossEntropyLoss().to(device)#.cuda()

    cudnn.benchmark = True

    # data loading code
    if args.dataset == 'cifar10':
        transform_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if args.auto_augment:
            transform_train.append(AutoAugment())
        if args.cutout:
            transform_train.append(Cutout())
        transform_train.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_train = transforms.Compose(transform_train)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])

        train_set = datasets.CIFAR10(
            root='~/data',
            train=True,
            download=True,
            transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=128,
            shuffle=True,
            num_workers=8)

        test_set = datasets.CIFAR10(
            root='~/data',
            train=False,
            download=True,
            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=128,
            shuffle=False,
            num_workers=8)

        num_classes = 10

    elif args.dataset == 'cifar100':
        transform_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if args.auto_augment:
            transform_train.append(AutoAugment())
        if args.cutout:
            transform_train.append(Cutout())
        transform_train.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_train = transforms.Compose(transform_train)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        train_set = datasets.CIFAR100(
            root='~/data',
            train=True,
            download=True,
            transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=128,
            shuffle=True,
            num_workers=8)

        test_set = datasets.CIFAR100(
            root='~/data',
            train=False,
            download=True,
            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=128,
            shuffle=False,
            num_workers=8)

        num_classes = 100

    elif args.dataset == 'tinyimagenet':
        # transform_train = [
        #      transforms.RandomRotation(20),
        #      transforms.RandomHorizontalFlip(0.5)]
        transform_train =[
        #    transforms.RandomCrop(32, padding=4),
           transforms.RandomHorizontalFlip(),
        ]

        transform_train=[]
        if args.auto_augment:
            transform_train.append(AutoAugment())
        if args.cutout:
            transform_train.append(Cutout())
        transform_train.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975),
                                 (0.2302, 0.2265, 0.2262)),
        ])
        transform_train = transforms.Compose(transform_train)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975),
                                 (0.2302, 0.2265, 0.2262)),
        ])

        full_train_set = datasets.ImageFolder(root='/home/csgrad/dl_228/tiny-imagenet-200/train', transform=transform_train)
        test_set = datasets.ImageFolder(root='/home/csgrad/dl_228/tiny-imagenet-200/val', transform=transform_test)

        class_counts = {i:0 for i in range(200)}
        max_samples_per_class = 250

        selected_indices=[]
        for idx, label in enumerate(full_train_set.targets):
          if class_counts[label]<max_samples_per_class:
            selected_indices.append(idx)
            class_counts[label]+=1
          if all(count == max_samples_per_class for count in class_counts.values()):
            break
        
        train_set = torch.utils.data.Subset(full_train_set, selected_indices)

        B = 128

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=B, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=B, shuffle=False, num_workers=4)

        num_classes = 200

    # create model
    # model = WideResNet(args.depth, args.width, num_classes=num_classes)
    # model = model.cuda()
    # model = torchvision.models.resnet50(weights=None)
    model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
    for param in model.parameters():
         param.requires_grad = False
    for param in model.classifier[6].parameters():  # Last layer in classifier
         param.requires_grad = True
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # model.apply(init_weights)
    model.to(device)
    # model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
    #         momentum=args.momentum, weight_decay=args.weight_decay)

    # scheduler = lr_scheduler.MultiStepLR(optimizer,
    #         milestones=[int(e) for e in args.milestones.split(',')], gamma=args.gamma)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'acc', 'val_loss', 'val_acc'
    ])

    train_loss_history=[]
    train_acc_history=[]
    val_acc_history=[]
    best_acc = 0
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' %(epoch+1, args.epochs))

        # scheduler.step()

        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        val_log = validate(args, test_loader, model, criterion)

        print('loss %.4f - acc %.4f - val_loss %.4f - val_acc %.4f'
            %(train_log['loss'], train_log['acc'], val_log['loss'], val_log['acc']))
        train_loss_history.append(train_log['loss']/100.0)
        train_acc_history.append(train_log['acc']/100.0)
        val_acc_history.append(val_log['acc']/100.0)
    

        if val_log['acc'] > best_acc:
            torch.save(model.state_dict(), 'models/%s/model.pth' %args.name)
            best_acc = val_log['acc']
            print("=> saved best model")

    # Define the data to store
    data = {
        "train_loss_history": train_loss_history,  # Replace with your train loss values
        "train_acc_history": train_acc_history,   # Replace with your train accuracy values
        # "val_loss": val_log["loss"],     # Replace with your validation loss values
        "val_acc_history": val_acc_history,       # Replace with your validation accuracy values
    }    # Define the directory to save the `.pkl` file
    output_dir = "models/"
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist    # Save the data to a `.pkl` file
    output_path = os.path.join(output_dir, f"AutoAugment_VGG16_Adam_metrics.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(data, f)    
        
    print(f"Metrics saved to {output_path}")

if __name__ == '__main__':
    main()
