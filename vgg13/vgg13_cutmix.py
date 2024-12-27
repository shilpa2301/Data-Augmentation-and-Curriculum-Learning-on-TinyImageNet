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

def train_model_cutmix(model, criterion, optimizer, train_loader, test_loader, epochs, device, alpha=1.0, patience=10, min_delta=0.0, p=0.5):

    epoch_loss=[]
    epoch_acc=[]
    test_acc=[]
    test_loss=[]

    best_val_loss = float('inf')
    best_val_acc = float('-inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
      print()

      running_loss=0.0
      running_corrects=0

      model.train()
      total_labels=0

      
      for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # print(inputs.shape[0])
        optimizer.zero_grad()

        r=np.random.rand(1)
        if r<p:
          lambda_val = np.random.beta(alpha, alpha)
          rand_index = torch.randperm(inputs.shape[0]).to(device)
          target_a = labels
          target_b = labels[rand_index]
          bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.shape, lambda_val)
          inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
          # adjust lambda to exactly match pixel ratio
          lambda_val = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.shape[-1] * inputs.shape[-2]))
          outputs=model(inputs)
          loss= mixcut_criterion(criterion, outputs, target_a, target_b, lambda_val)
        else:
           
          outputs=model(inputs)
          loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        _,predicted=outputs.max(1)
        if r<p:
          running_corrects += (lambda_val * predicted.eq(target_a).sum().item()
                      + (1 - lambda_val) * predicted.eq(target_b).sum().item())
          total_labels += len(target_a)+len(target_b)
          
        else:
          running_corrects +=  predicted.eq(labels).sum().item()
          total_labels += len(labels)
          # epoch_loss.append(running_loss/len(train_loader.dataset))

      epoch_loss.append(running_loss/len(train_loader.dataset))
      epoch_acc.append(running_corrects/total_labels)

      #evaluate on test set
      model.eval()
      test_corrects=0
      total_labels=0
      running_loss=0
      with torch.no_grad():
        for inputs, labels in test_loader:
          inputs, labels = inputs.to(device), labels.to(device)
          outputs=model(inputs)
          loss= criterion(outputs, labels)
          running_loss += loss.item()

          _,predicted=outputs.max(1)
          test_corrects+=predicted.eq(labels).sum().item()
          total_labels += len(labels)

        epoch_test_loss = running_loss / len(test_loader.dataset)
        epoch_test_acc = test_corrects/total_labels
        test_loss.append(epoch_test_loss)
        test_acc.append(epoch_test_acc)

      print(f"Epoch: {epoch+1}, Loss: {epoch_loss[-1]:.4f}, Train Acc: {epoch_acc[-1]:.4f},Test Loss:{test_loss[-1]:.4f}, Test Acc: {test_acc[-1]:.4f}")

      # Check for improvement based on val loss
      # if epoch_test_loss < best_val_loss - min_delta:
      #     best_val_loss = epoch_test_loss
      #     best_acc = epoch_test_acc
      #     # best_model_wts = copy.deepcopy(model.state_dict())
      #     epochs_no_improve = 0  # Reset counter
      # else:
      #     epochs_no_improve += 1
      #     print(f"Patience = {epochs_no_improve}/{patience}")

      # # Early stopping
      # if epochs_no_improve >= patience:
      #     print(f"Early stopping triggered after {epoch + 1} epochs without improvement.")
      #     break

      #   if test_acc[-1]>best_acc:
      #     best_acc=test_acc[-1]
      #     best_model_state = model.state_dict()
      #     print(f"Best model state stored with accuracy: {best_acc:.4f}")

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

    

    alpha_set=[0.2, 0.4, 0.6]
    train_loss_alpha=[]
    test_acc_alpha = []
    train_acc_alpha=[]

    for alpha in alpha_set:
    #   model = torchvision.models.resnet18(weights=None)
    #   model = torchvision.models.resnet34(weights=None)
      model = torchvision.models.vgg13(weights=torchvision.models.VGG13_Weights.IMAGENET1K_V1)
      for param in model.parameters():
           param.requires_grad = False
      for param in model.classifier[6].parameters():  # Last layer in classifier
           param.requires_grad = True

      criterion = nn.CrossEntropyLoss()
      optimizer = optim.Adam(model.parameters(), lr=0.001)
      # model.apply(init_weights)
      model.to(device)
      scheduler=None
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1,
    #             momentum=0.9, weight_decay=5e-4)
    # milestones = '60,120,160'
    # scheduler = lr_scheduler.MultiStepLR(optimizer,
    #             milestones=[int(e) for e in milestones.split(',')], gamma=0.2)


      print('*'*100)
      print(f"Alpha: {alpha}")
      model, train_loss, train_acc, test_acc = train_model_cutmix(model,  criterion, optimizer, train_loader, 
                                                                  test_loader, epochs, device,  alpha=1.0, 
                                                                  patience=10, min_delta=0.0, p=0.5)
      # evaluate_model(model, test_loader)
      train_loss_alpha.append(train_loss)
      test_acc_alpha.append(test_acc)
      train_acc_alpha.append(train_acc)


    data_to_save = {
            "model": model,
            "train_loss_history": train_loss_alpha[0],
            # "val_loss_history": train_loss_vanilla,
            "train_acc_history": train_acc_alpha[0],
            "val_acc_history": test_acc_alpha[0]
        }

        # Save the dictionary to a pickle file
    with open("data_pkl/vgg13_cutmix_0.2_data.pkl", "wb") as f:
                pickle.dump(data_to_save, f)
    
    
    data_to_save = {
                "model": model,
                "train_loss_history": train_loss_alpha[1],
                # "val_loss_history": train_loss_vanilla,
                "train_acc_history": train_acc_alpha[1],
                "val_acc_history": test_acc_alpha[1]
            }
    
            # Save the dictionary to a pickle file
    with open("data_pkl/vgg13_cutmix_0.4_data.pkl", "wb") as f:
                pickle.dump(data_to_save, f)
    
    data_to_save = {
                "model": model,
                "train_loss_history": train_loss_alpha[2],
                # "val_loss_history": train_loss_vanilla,
                "train_acc_history": train_acc_alpha[2],
                "val_acc_history": test_acc_alpha[2]
            }  
  
               
    # Save the dictionary to a pickle file
    with open("data_pkl/vgg13_cutmix_0.6_data.pkl", "wb") as f:
            pickle.dump(data_to_save, f)

if __name__ == '__main__':
    main()