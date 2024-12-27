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
import time

parser = argparse.ArgumentParser(description='PyTorch TinyImageNet Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('-e', '--epochs', default=25, type=int,
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

def plot_metrics(metrics, group_cut):
    # Plot training loss curves
    plt.figure(figsize=(10, 6))
    for group_idx in range(args.group_num):
        plt.plot(metrics['train_loss'][group_idx], label=f'Group {group_idx + 1}')
    plt.title('Training Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig('plots_batch/LGL training_loss_curves.png')  # Save the figure
    # plt.show()

    # Plot training accuracy curves
    plt.figure(figsize=(10, 6))
    for group_idx in range(args.group_num):
        plt.plot(metrics['train_acc'][group_idx], label=f'Group {group_idx + 1}')
    plt.title('Training Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()
    plt.savefig('plots_batch/LGL training_accuracy_curves.png')  # Save the figure
    # plt.show()

    # Plot test accuracy curves
    plt.figure(figsize=(10, 6))
    for group_idx in range(args.group_num):
        plt.plot(metrics['test_acc'][group_idx], label=f'Group {group_idx + 1}')
    plt.title('Test Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()
    plt.savefig('plots_batch/LGL test_accuracy_curves.png')  # Save the figure
    # plt.show()


def resume_checkpoint_group(net,resume_path):
    print("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(resume_path)
    net.load_state_dict(checkpoint['state_dict'])

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
    for group_idx in metrics["train_loss"].keys():
        data = {
            "train_loss_history": metrics["train_loss"][group_idx],  # Train loss for this group
            "train_acc_history": metrics["train_acc"][group_idx],   # Train accuracy for this group
            "val_acc_history": metrics["test_acc"][group_idx],      # Test accuracy for this group
        }

        # Create a unique filename for each group
        output_filename = f"metrics_group_{group_idx + 1}.pkl"
        output_path = os.path.join(output_dir, output_filename)

        # Save the data to a `.pkl` file
        with open(output_path, "wb") as f:
            pickle.dump(data, f)

        print(f"Metrics for group {group_idx + 1} saved to {output_path}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Initialize dictionaries to store metrics for each group
metrics = {
    'train_loss': {i: [] for i in range(args.group_num)},
    'train_acc': {i: [] for i in range(args.group_num)},
    'test_acc': {i: [] for i in range(args.group_num)},
}

#data augmentation
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mixcut_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Training
def train(epoch, group_idx):
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

        #cutmix
        alpha=1.0
        p=0.5
        r=np.random.rand(1)
        if r<p:
          lambda_val = np.random.beta(alpha, alpha)
          rand_index = torch.randperm(inputs.shape[0]).to(device)
          target_a = targets
          target_b = targets[rand_index]
          bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.shape, lambda_val)
          inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
          # adjust lambda to exactly match pixel ratio
          lambda_val = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.shape[-1] * inputs.shape[-2]))
          outputs=net(inputs)
          loss= mixcut_criterion(criterion, outputs, target_a, target_b, lambda_val)
        else:
           
          outputs=net(inputs)
          loss = criterion(outputs, targets)

        # outputs = net(inputs)
        # loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        
        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()
        if r<p:
          correct += (lambda_val * predicted.eq(target_a).sum().item()
                      + (1 - lambda_val) * predicted.eq(target_b).sum().item())
          total += len(target_a)+len(target_b)
          
        else:
          correct +=  predicted.eq(targets).sum().item()
          total += len(targets)

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    # Store metrics for the current epoch
    metrics['train_loss'][group_idx].append(train_loss / len(trainloader))
    metrics['train_acc'][group_idx].append(100. * correct / total)
    

def test(epoch, group_idx):
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
    metrics['test_acc'][group_idx].append(acc)

    if acc > best_acc:
        print('Saving..')
        state = {
            'state_dict': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('Checkpoint_logger_batch'):
            os.mkdir('Checkpoint_logger_batch')
        torch.save(state, os.path.join(args.save_dir, 'vgg13_model_best.pth.tar'))
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
dataset_group_num = args.group_num
group_cut = np.zeros([args.group_num])
for i in range(args.group_num):
    group_cut[i] = math.ceil(dataset_labels_num/float(dataset_group_num))
    dataset_labels_num = dataset_labels_num - int(group_cut[i])
    dataset_group_num = dataset_group_num -1
group_cut = group_cut.astype(np.int64)
print(f"group_cut={group_cut}")


start_time=time.time()
for i in range(args.group_num):
    best_acc = 0
    start_epoch = 1

    # Get the list of all class labels (folder names) in the dataset directory
    all_class_labels = sorted(os.listdir(os.path.join(args.data_dir,'train')))  # Assuming the dataset directory contains class folders
    # print(f"all_class_albels={all_class_labels}")
    # print(f"All class labels: {all_class_labels}")

    if i == 0:
        # Randomly sample class labels for the first group
        clusters_chosen = np.array(random.sample(all_class_labels, group_cut[i]))
        used_clusters = clusters_chosen
        print(f"Used clusters (group {i+1}): {len(used_clusters)}")
    else:
        # Randomly sample additional class labels for subsequent groups
        remaining_classes = [cls for cls in all_class_labels if cls not in used_clusters]
        clusters_chosen = np.array(random.sample(remaining_classes, group_cut[i]))
        used_clusters = np.append(used_clusters, clusters_chosen)
        print(f"Used clusters (group {i+1}): {len(used_clusters)}")

    # Print the dataset directory and its type for debugging
    print("args.data_dir:", args.data_dir)
    print("Type of args.data_dir:", type(args.data_dir))

    # Create the output directory for the current group
    output_data_dir = os.path.join(
        '/home/csgrad/dl_228/LGL/Local-to-Global-Learning-for-DNNs/Datasets_batch',
        f'group_{i+1}'
    )
    print(f"Output data directory for group {i+1}: {output_data_dir}")

    # You can now use `clusters_chosen` to process or copy the data for the selected class labels
    for cluster in clusters_chosen:
        class_folder_path = os.path.join(args.data_dir, cluster)
        # print(f"Processing data for class: {cluster}, path: {class_folder_path}")

    ds_tiny.TinyImageNetDataset.create_dataset_group(
        args.data_dir,       # original_data_dir
        output_data_dir,     # output_data_dir (must be a string)
        used_clusters,       # clusters to use (numpy array)
        i+1,                 # group_order
        args.group_num       # total number of groups
    )
    print(f"dataset group created")

    

    train_transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5, 0.5])
        ])

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5, 0.5])
        ])

    train_dir_for_idx='/home/csgrad/dl_228/tiny-imagenet-200/train'

    train_data = ds_tiny.TinyImageNetDataset(data_dir=args.data_dir, split='train', 
                                             transform=train_transform, used_cluster=used_clusters, train_dir=train_dir_for_idx)
    test_data = ds_tiny.TinyImageNetDataset(data_dir=args.data_dir, split='val', 
                                            transform=test_transform, used_cluster=used_clusters, train_dir=train_dir_for_idx)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size =  args.batch_size, shuffle = True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_data, batch_size =  100, shuffle = False, num_workers=0)
    print(f"trainloader={len(trainloader.dataset)}")
    if i==0:
        # net.classifier = nn.Linear(int(2048),int(sum(group_cut[:i+1])))
        net.classifier = nn.Linear(int(2048),len(used_clusters))
        print(group_cut[:i+1])
    else:
        resume_path = args.save_dir + '/vgg13_model_best.pth.tar'
        resume_checkpoint_group(net, resume_path)
        params = net.state_dict()
        weight = params['classifier.weight']
        width,height = weight.shape
        bias = params['classifier.bias']

        # net.classifier = nn.Linear(int(2048),int(sum(group_cut[:i+1])))
        net.classifier = nn.Linear(int(2048),len(used_clusters))

        params_new = net.state_dict()
        weight_new = params_new['classifier.weight']
        bias_new = params_new['classifier.bias']
        weight_new[:width,:] = weight
        bias_new[:width] = bias
        net.load_state_dict(params_new)

    for epoch in range(start_epoch, args.epochs):
        train(epoch,i)
        test(epoch,i)

    if i == args.group_num - 1:
        print(best_acc)

end_time=time.time()
total_training_time = end_time - start_time
print (f"time taken in vanilla = {total_training_time}")

# After training all groups
save_metrics_per_group(metrics, output_dir="data_pkl_batch")
plot_metrics(metrics, group_cut)



















