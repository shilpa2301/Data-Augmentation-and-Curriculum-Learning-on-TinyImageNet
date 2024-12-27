import os
import urllib.request
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import copy
import numpy as np
from torch.nn.utils import clip_grad_norm_
import torch.multiprocessing as mp
import sys
from torchvision.models import ResNet18_Weights, ResNet34_Weights, VGG16_Weights, MobileNet_V2_Weights, GoogLeNet_Weights
from tqdm import tqdm
import random
import torch.nn.functional as F
from cvae_run import cVAE
from torchvision.models.vision_transformer import VisionTransformer, ViT_B_16_Weights, vit_b_16
# import timm
from modules import UNet_conditional
from utils import plot_images, save_images
from ddpm_conditional import Diffusion
import pickle



mp.set_start_method('spawn', force=True)

# Ensure GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_balanced_subset(dataset, num_classes=200, samples_per_class=100, seed=42):
    """
    Create a balanced subset of the dataset with a fixed number of samples per class.
    
    Args:
        dataset (Dataset): The original dataset (e.g., ImageFolder).
        num_classes (int): Total number of classes in the dataset.
        samples_per_class (int): Number of samples to select per class.
        seed (int): Random seed for reproducibility.
        
    Returns:
        Subset: A subset of the original dataset containing the selected samples.
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Group indices by class
    class_indices = {i: [] for i in range(num_classes)}  # Dictionary to hold indices for each class
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    # Select samples_per_class samples from each class
    selected_indices = []
    for label, indices in class_indices.items():
        if len(indices) < samples_per_class:
            raise ValueError(f"Class {label} has fewer than {samples_per_class} samples.")
        selected_indices.extend(random.sample(indices, samples_per_class))
    
    # Create a subset of the dataset
    return Subset(dataset, selected_indices)

# Modify the load_data function to include the new train subset
def load_data_with_balanced_subset(batch_size=64, data_augmentation=False, samples_per_class=100):
    if data_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262]), 
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262]), 
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262]), 
    ])

    # Load the original datasets
    train_dataset = datasets.ImageFolder(root='/home/csgrad/dl_228/tiny-imagenet-200/train', transform=train_transform)
    val_dataset = datasets.ImageFolder(root='/home/csgrad/dl_228/tiny-imagenet-200/val', transform=test_transform)
    test_dataset = datasets.ImageFolder(root='/home/csgrad/dl_228/tiny-imagenet-200/test', transform=test_transform)

    # Create a balanced subset for training
    balanced_train_dataset = create_balanced_subset(train_dataset, num_classes=200, samples_per_class=samples_per_class)

        # # Visualize 5 random images from the training dataset
    # visualize_random_images(train_dataset, title="Random Training Images")
    
    # # Visualize 5 random images from the test dataset
    # visualize_random_images(test_dataset, title="Random Test Images")

    # Create data loaders
    train_loader = DataLoader(balanced_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader
 
def visualize_random_images(dataset, title):
    indices = random.sample(range(len(dataset)), 5)  # Randomly select 5 indices
    images = [dataset[i][0].permute(1, 2, 0).numpy() for i in indices]  # Get images and convert to numpy
    labels = [dataset[i][1] for i in indices]  # Get labels

    plt.figure(figsize=(10, 5))
    plt.suptitle(title)

    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
        plt.title(f"Label: {labels[i]}")

    plt.tight_layout()
    plt.show()

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# Function to get model architectures with Dropout
def get_model(model_name, device='cuda'):
    if model_name == 'resnet18':
        model = models.resnet18(weights=False)

        # model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 200)  # Single linear layer
        model.apply(init_weights)
    elif model_name == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 200)  # Single linear layer
    elif model_name == 'vgg':
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 200)  # Single linear layer
    elif model_name == 'mobilenet':
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 200)  # Single linear layer
    elif model_name == 'googlenet':
        model = models.googlenet(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 200)  # Single linear layer
    else:
        raise ValueError("Invalid model name. Choose from 'resnet', 'vgg', 'mobilenet', 'googlenet'")
    
    # # Freeze all layers
    # for param in model.parameters():
    #     param.requires_grad = False

    # # Unfreeze the parameters of the last layer based on model architecture
    # if model_name in ['resnet18', 'resnet34', 'googlenet']:
    #     for param in model.fc.parameters():
    #         param.requires_grad = True
    # elif model_name == 'vgg':
    #     for param in model.classifier[6].parameters():  # Last layer in classifier
    #         param.requires_grad = True
    # elif model_name == 'mobilenet':
    #     for param in model.classifier[1].parameters():  # Last layer in classifier
    #         param.requires_grad = True

    return model.to(device)

# Function to train and evaluate the model
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, 
                num_epochs=10, l1_lambda=0.001, l2_lambda=0.01, max_norm=None, patience=5, 
                min_delta=0.01, unet_model=None, diff_net=None):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_val_loss = float('inf')
    best_val_acc = float('-inf')
    epochs_no_improve = 0

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print("-" * 50)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        # running_corrects = 0
        total,total_tr=0,0
        correct, correct_tr=0,0
        

        for inputs, labels in tqdm(train_loader, desc="Training", unit="batch"):

           
            inputs, labels = inputs.to(device), labels.to(device)
            # Decide whether to sample this batch (10% probability)
            if random.random() < 0.1:  # 10% chance
                print("Sampling this batch...")
                batch_size = inputs.size(0)
                num_to_replace = max(1, int(batch_size * 0.05))  # Replace 5% of the batch, at least 1 image

                # Randomly select indices to replace
                indices_to_replace = random.sample(range(batch_size), num_to_replace)

                for idx, idx_to_replace in enumerate(indices_to_replace):
                    # print(f"Image {idx + 1}/{num_to_replace} getting sampled")
                    target_class = labels[idx_to_replace].item()  # Get the label at the selected index
                    sampled_labels = torch.Tensor([target_class]).long().to(device)  # Single label
                    sampled_images = diff_net.sample(unet_model, n=1, labels=sampled_labels)  # Generate the sampled image
                    # print(f"Replacing image at index {idx_to_replace} with sampled image of class {target_class}")

                    # Replace the image and label in the batch
                    inputs[idx_to_replace] = sampled_images[0]  # Replace the image
                    labels[idx_to_replace] = sampled_labels[0]  # Replace the label
            # else:
                # print("Using original batch...")

            optimizer.zero_grad()

            # outputs = model(inputs)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


            running_loss += loss.item() 
            _, predicted_tr = outputs.max(1)
            # print(f"predicted_tr={predicted_tr}")
            total_tr += labels.size(0)

            correct_tr += predicted_tr.eq(labels).sum().item()
            # print(f"correct_tr={correct_tr}")


        epoch_loss = running_loss / len(train_loader.dataset)
        # print(f"correct_tr, total_tr={correct_tr}, {total_tr}")
        epoch_acc = correct_tr / total_tr
        

        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation phase
        model.eval()
        running_loss = 0.0
        # running_corrects = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation", unit="batch"):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                # running_corrects += torch.sum(preds == labels.data)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                # print(f"val correct predicted={predicted.eq(labels).sum().item()}")
                correct += predicted.eq(labels).sum().item()

        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total


        val_loss_history.append(epoch_val_loss)
        val_acc_history.append(epoch_val_acc)

        print(f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

        # Check for improvement based on val loss
        if epoch_val_loss < best_val_loss - min_delta:
            best_val_loss = epoch_val_loss
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0  # Reset counter
        else:
            epochs_no_improve += 1
            print(f"Patience = {epochs_no_improve}/{patience}")

        # # Check for improvement in validation accuracy
        # if epoch_val_acc > best_val_acc + min_delta:
        #     best_val_acc = epoch_val_acc
        #     best_acc = epoch_acc
        #     best_model_wts = copy.deepcopy(model.state_dict())
        #     epochs_no_improve = 0  # Reset counter
        # else:
        #     epochs_no_improve += 1

        # Step the scheduler
        if scheduler: 
            scheduler.step()

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs without improvement.")
            break

        plt.figure(figsize=(12, 5))
        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(train_loss_history, label="Train Loss")
        plt.plot(val_loss_history, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss over Epochs")
        
        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(train_acc_history, label="Train Accuracy")
        plt.plot(val_acc_history, label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy over Epochs")

        plt.tight_layout()  # Adjust layout to prevent overlapping
        plt.savefig("plots/Running_batch_job_resdiff_metrics_3.png", dpi=300, bbox_inches='tight') 

        plt.show()

        # Create a dictionary to store all the variables
        data_to_save = {
            "model": model,
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
            "train_acc_history": train_acc_history,
            "val_acc_history": val_acc_history
        }

        # Save the dictionary to a pickle file
        with open("data_pkl/resnet18_diffusion_aug_data_3.pkl", "wb") as f:
            pickle.dump(data_to_save, f)


        
    print(f"Best Val Acc: {best_acc:.4f}")

    # Load best model weights
    # model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history


# Main function
if __name__ == "__main__":
    torch.cuda.empty_cache()

    
    diff_net = UNet_conditional(num_classes=200).to(device)
    ckpt_path = "diff_models/DDPM_conditional/ema_ckpt.pt"  # Path to the saved checkpoint
    diff_net.load_state_dict(torch.load(ckpt_path))
    diff_net.eval()
    diffusion = Diffusion(img_size=64, device=device)

    # Step 2: Load data
    B=256
    lr=0.001
    train_loader, val_loader, test_loader = load_data_with_balanced_subset(batch_size=B, data_augmentation=False)

    # Step 3: Get model
    model_name = 'resnet18'
    print(f"Model Used = {model_name}")
    model = get_model(model_name)

    # Step 4: Define loss function and optimizer    
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=lr) #, betas=(0.9, 0.999))

    # Step 5: Initialize LR scheduler
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # Step 6: Train and evaluate the model
    trained_model, train_loss_vanilla_resnet, val_loss_vanilla_resnet, train_acc_vanilla_resnet, val_acc_vanilla_resnet = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=1000, 
        l1_lambda=0.0, l2_lambda=0.0,
        max_norm=None, min_delta=0.0, patience=20,
        unet_model=diff_net, diff_net=diffusion
    )

    data_to_save = {
                "model": model,
                "train_loss_history": train_loss_vanilla_resnet,
                # "val_loss_history": train_loss_vanilla,
                "train_acc_history": train_acc_vanilla_resnet,
                "val_acc_history": val_acc_vanilla_resnet
            }

            # Save the dictionary to a pickle file
    with open("data_pkl/resnet18_diffusion_aug_data_final_3.pkl", "wb") as f:
                pickle.dump(data_to_save, f)

     # Step 7: Plot training and validation loss and accuracy
    plt.figure(figsize=(12, 5))
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_vanilla_resnet, label="Train Loss")
    plt.plot(val_loss_vanilla_resnet, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over Epochs")
    # Mark the lowest validation loss with a red dashed line
    min_val_loss = min(val_loss_vanilla_resnet)
    min_val_loss_index = val_loss_vanilla_resnet.index(min_val_loss)
    plt.axvline(x=min_val_loss_index, color='red', linestyle='--', label='Lowest Val Loss')

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_vanilla_resnet, label="Train Accuracy")
    plt.plot(val_acc_vanilla_resnet, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy over Epochs")
    
    # Mark the highest validation accuracy with a red dashed line
    max_val_acc= max(val_acc_vanilla_resnet)
    max_val_acc_index = val_acc_vanilla_resnet.index(max_val_acc)
    plt.axvline(x=max_val_acc_index, color='red', linestyle='--', label='Highest Val Accuracy')

    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.savefig("plots/training_batch_job_resdiff_metrics_3.png", dpi=300, bbox_inches='tight') 

    plt.show()

   
