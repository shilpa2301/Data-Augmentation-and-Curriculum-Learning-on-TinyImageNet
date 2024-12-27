import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset

import argparse
import matplotlib.pyplot as plt
import numpy as np
import copy
import math
import pickle
from torch.optim import lr_scheduler


device='cuda' if torch.cuda.is_available() else 'cpu'
# device

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
def train_model(model, criterion, optimizer, train_loader, test_loader, 
                epochs, device, scheduler=None):

    epoch_loss=[]
    epoch_acc=[]
    test_acc=[]
    best_acc=0.0
    best_model_state = None

    for epoch in range(epochs):
      print()
      # print('-'*50)
      # print(f"Epoch: {epoch+1}/{epochs}")
      # print('-'*50)
      running_loss=0.0
      running_corrects=0
      if scheduler:
        scheduler.step

      model.train()
      total_labels=0
      for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs=model(inputs)
        loss= criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        _,predicted=outputs.max(1)
        running_corrects+=predicted.eq(labels).sum().item()
        total_labels += len(labels)

      epoch_loss.append(running_loss/len(train_loader.dataset))
      epoch_acc.append(running_corrects/total_labels)

      #evaluate on test set
      model.eval()
      # test_loss=0.0
      test_corrects=0
      total_labels=0
      with torch.no_grad():
        for inputs, labels in test_loader:
          inputs, labels = inputs.to(device), labels.to(device)
          outputs=model(inputs)
          loss= criterion(outputs, labels)
          # test_loss+=loss.item()
          _,predicted=outputs.max(1)
          test_corrects+=predicted.eq(labels).sum().item()
          total_labels += len(labels)

        test_acc.append(test_corrects/total_labels)

      print(f"Epoch: {epoch+1}, Loss: {epoch_loss[-1]:.4f}, Train Acc: {epoch_acc[-1]:.4f}, Test Acc: {test_acc[-1]:.4f}")

      # if test_acc[-1]>best_acc:
      #   best_acc=test_acc[-1]
      #   # torch.save(model.state_dict(), 'best_model.pth')
      #   best_model_state = model.state_dict()
      #   print(f"Best model state stored with accuracy: {best_acc:.4f}")

    # model.load_state_dict(best_model_state)
    return model, epoch_loss, epoch_acc, test_acc

def main():
    train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262]), 
            ])

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262]), 
        ])


        # Load the original datasets
    full_train_set = torchvision.datasets.ImageFolder(root='/home/csgrad/dl_228/tiny-imagenet-200/train', transform=train_transform)
    test_set = torchvision.datasets.ImageFolder(root='/home/csgrad/dl_228/tiny-imagenet-200/val', transform=test_transform)
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

    train_set = Subset(full_train_set, selected_indices)

    B = 128 #batch size
    lr = 0.001 #learning rate
    epochs = 40 #number of epochs

    train_loader = DataLoader(train_set, batch_size=B, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=B, shuffle=False, num_workers=2)

    print(f"Total train_samples: {len(train_set)}")
    print(f"Total test_samples: {len(test_set)}")


    # model = torchvision.models.resnet18(weights=None)
    model = torchvision.models.resnet18(weights=None)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler=None
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1,
    #             momentum=0.9, weight_decay=5e-4)
    # milestones = '60,120,160'
    # scheduler = lr_scheduler.MultiStepLR(optimizer,
    #             milestones=[int(e) for e in milestones.split(',')], gamma=0.2)


    model.apply(init_weights)
    model.to(device)


    model, train_loss_vanilla, train_acc_vanilla, test_acc_vanilla = train_model(model,  criterion, optimizer,
                                                                                 train_loader, test_loader, epochs, 
                                                                                 device, scheduler=scheduler)
    # evaluate_model(model, test_loader)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_vanilla, label='Train Loss', color='blue')
    # plt.plot(test_acc, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_vanilla, label='Train Accuracy', color='blue')
    plt.plot(test_acc_vanilla, label='Test Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Add a title to the entire figure
    plt.suptitle('Training Loss and Accuracy (Vanilla Resnet50)')

    # Save the plot to a file
    plt.savefig('plots/vanilla_resnet18_Adam.png', dpi=300, bbox_inches='tight')

    data_to_save = {
            "model": model,
            "train_loss_history": train_loss_vanilla,
            # "val_loss_history": train_loss_vanilla,
            "train_acc_history": train_acc_vanilla,
            "val_acc_history": test_acc_vanilla
        }

    # Save the dictionary to a pickle file
    with open("data_pkl/resnet18_vanilla_data.pkl", "wb") as f:
            pickle.dump(data_to_save, f)

if __name__ == '__main__':
    main()