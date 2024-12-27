# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np
from torch.autograd import Variable
from models import *
from utils import progress_bar
from selection_strategy import clusters_chosen_random
# from dataset_class import *
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import dataset_class_tinyimagenet as ds_tiny
import matplotlib.pyplot as plt
import os
import pickle
# from dataset_class_tinyimagenet.TinyImageNetDataset import create_dataset_group, dataset_train_group


parser = argparse.ArgumentParser(description='PyTorch TinyImageNet Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('-e', '--epochs', default=8, type=int,
                    help='number of total epochs (default: 200)')
parser.add_argument('--save-dir', default='Checkpoint_logger_batch', type=str,
                    help='directory of saved model (default: model/saved)')
parser.add_argument('--data-dir', default='/home/csgrad/dl_228/tiny-imagenet-200/', type=str,
                    help='directory of training/testing data (default: datasets)')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

parser.add_argument('--group-num', default=2, type = int, 
                    help='the num of pre-train')

parser.add_argument('--cluster-num', default=200, type = int, 
                    help='cluster number')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Model
print('==> Building model..')
# net = VGG('VGG16')
# net = ResNet18()
net = VGG('VGG13')
#net = net.to(device)
#if device == 'cuda':
    #net = torch.nn.DataParallel(net)
    #cudnn.benchmark = True
use_cuda = torch.cuda.is_available()
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

def plot_metrics(metrics):
    # Plot training loss curves
    plt.figure(figsize=(10, 6))
    # for group_idx in range(args.group_num):
    plt.plot(metrics['train_loss'], label=f'Train_loss')
    plt.title('Training Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig('plots/Vanilla LGL training_loss_curves local.png')  # Save the figure
    # plt.show()

    # Plot training accuracy curves
    plt.figure(figsize=(10, 6))
    # for group_idx in range(args.group_num):
    plt.plot(metrics['train_acc'], label=f'Train_Acc')
    plt.title('Training Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()
    plt.savefig('plots/Vanilla LGL training_accuracy_curves local.png')  # Save the figure
    # plt.show()

    # Plot test accuracy curves
    plt.figure(figsize=(10, 6))
    # for group_idx in range(args.group_num):
    plt.plot(metrics['test_acc'], label=f'Test Acc')
    plt.title('Test Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()
    plt.savefig('plots/Vanilla LGL test_accuracy_curves local.png')  # Save the figure
    # plt.show()


def save_metrics_per_group(metrics, output_dir="data_pkl"):
    """
    Save the metrics for each group as separate pickle files.

    Args:
        metrics (dict): Dictionary containing train_loss, train_acc, and test_acc.
        output_dir (str): Directory to save the pickle files.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics for each group
    # for group_idx in metrics["train_loss"].keys():
    #     data = {
    #         "train_loss_history": metrics["train_loss"][group_idx],  # Train loss for this group
    #         "train_acc_history": metrics["train_acc"][group_idx],   # Train accuracy for this group
    #         "val_acc_history": metrics["test_acc"][group_idx],      # Test accuracy for this group
    #     }

    #     # Create a unique filename for each group
    #     output_filename = f"metrics_vanilla_VGG13.pkl"
    #     output_path = os.path.join(output_dir, output_filename)

    #     # Save the data to a `.pkl` file
    #     with open(output_path, "wb") as f:
    #         pickle.dump(data, f)
    data = {
        "train_loss_history": metrics["train_loss"],
        "train_acc_history": metrics["train_acc"],
        "val_acc_history": metrics["test_acc"],
    }

    output_filename = f"metrics_vanilla_VGG13_local.pkl"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "wb") as f:
        pickle.dump(data, f)

       
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Initialize dictionaries to store metrics for each group
# metrics = {
#     'train_loss': {i: [] for i in range(1)},
#     'train_acc': {i: [] for i in range(1)},
#     'test_acc': {i: [] for i in range(1)},
# }

metrics = {
    'train_loss': [],
    'train_acc': [],
    'test_acc': [],
}

# Training
def train(epoch, trainloader):
    print('\nEpoch: %d' % epoch)
    lr = args.lr * pow(0.95,epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    net.train()
    if use_cuda:
        net.cuda()
    train_loss = 0
    correct = 0
    total = 0
    # running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = torch.FloatTensor(inputs), torch.LongTensor(targets)
        inputs, targets = Variable(inputs), Variable(targets)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    # Store metrics for the current epoch
    metrics['train_loss'].append(train_loss / len(trainloader))
    metrics['train_acc'].append(100. * correct / total)
    

def test(epoch, testloader):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = torch.FloatTensor(inputs), torch.LongTensor(targets)
            inputs, targets = Variable(inputs), Variable(targets)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            # print(f"inputs and targets shape={inputs.shape}, {targets.shape}")
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    metrics['test_acc'].append(acc)

    if acc > best_acc:
        print('Best Acc encountered..')
        best_acc = acc

def get_shape(nested_list):
    """
    Recursively calculates the shape of a nested list.

    Args:
        nested_list (list): The nested list to calculate the shape for.

    Returns:
        tuple: A tuple representing the shape of the nested list.
    """
    if isinstance(nested_list, list):
        # Recursively calculate the shape of the first element
        return (len(nested_list), *get_shape(nested_list[0]))
    else:
        # Base case: if not a list, return an empty tuple
        return ()


dataset_labels_num = 200

best_acc = 0
start_epoch = 1
# Get the list of all class labels (folder names) in the dataset directory
train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262]), 
            ])

test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262]), 
    ])
    # Load the original datasets
full_train_set = torchvision.datasets.ImageFolder(root='/home/csgrad/dl_228/LGL/Local-to-Global-Learning-for-DNNs/Datasets/group_5/train', transform=train_transform)
test_set = torchvision.datasets.ImageFolder(root='/home/csgrad/dl_228/LGL/Local-to-Global-Learning-for-DNNs/Datasets/group_5/val', transform=test_transform)
# test_dataset = torchvision.datasets.ImageFolder(root='/home/csgrad/dl_228/tiny-imagenet-200/test', transform=test_transform)
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

train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=0)
print(f"Total train_samples: {len(train_set)}")
print(f"Total test_samples: {len(test_set)}")

net.classifier = nn.Linear(int(2048),dataset_labels_num)
for epoch in range(start_epoch, args.epochs):
    train(epoch, train_loader)
    test(epoch, test_loader)

# After training all groups
save_metrics_per_group(metrics, output_dir="data_pkl")
plot_metrics(metrics)



















