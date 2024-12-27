# -*- coding: utf-8 -*-
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import shutil
from collections import defaultdict
import torchvision.transforms as transforms


class TinyImageNetDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, used_cluster=None, train_dir=None):
        """
        Args:
            data_dir (str): Path to the TinyImageNet dataset directory.
            split (str): Dataset split - 'train', 'val', or 'test'.
            transform (callable, optional): Transform to apply to the images.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.data = []
        self.labels = []
        self.used_cluster=used_cluster

        # classes = sorted(os.listdir(train_dir))
        # self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        classes = sorted(os.listdir(train_dir))
        self.global_class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        # Create a new mapping for used_cluster
        if used_cluster is not None:
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(used_cluster)}
        else:
            self.class_to_idx = self.global_class_to_idx

        if split == 'train':
            self._load_train_data()
        elif split == 'val':
            self._load_val_data()
        elif split == 'test':
            self._load_test_data()
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'")

    def _load_train_data(self):
        """
        Load training data for specified labels in used_cluster.

        Args:
            used_cluster (list): List of label names (classes) to load data for.
        """
        train_dir = os.path.join(self.data_dir, 'train')

        # Check if `used_cluster` is specified and not empty
        if self.used_cluster is not None and len(self.used_cluster) > 0:
            filtered_classes = self.used_cluster
        else:
            filtered_classes = sorted(os.listdir(train_dir))

        # classes = sorted(os.listdir(train_dir))
        class_to_idx=self.class_to_idx
        # class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        # print(f"used_cluster while loading train={filtered_classes}")

        for cls_name in filtered_classes:
            cls_folder = os.path.join(train_dir, cls_name, 'images')
            image_files = os.listdir(cls_folder)

            # Limit to 250 images per class
            random.shuffle(image_files)  # Shuffle to ensure randomness
            image_files = image_files[:250]  # Take the first 250 images

            for img_file in image_files:
                img_path = os.path.join(cls_folder, img_file)
                self.data.append(img_path)
                self.labels.append(class_to_idx[cls_name])

        print(f"images count= {len(self.data)}, {len(self.labels)}")

        # dataset = [(transform(Image.open(img_path).convert('RGB')), label) for img_path, label in zip(data, labels)]
        dataset = [self.data, self.labels]
        return dataset
    
    def _load_val_data(self):
        """
        Load validation data for specified labels in used_cluster.

        Args:
            used_cluster (list): List of label names (classes) to load data for.
        """
        val_dir = os.path.join(self.data_dir, 'val')

        # Check if `used_cluster` is specified and not empty
        if self.used_cluster is not None and len(self.used_cluster) > 0:
            filtered_classes = self.used_cluster
        else:
            filtered_classes = sorted(os.listdir(val_dir))
        # classes = sorted(os.listdir(val_dir))
        # class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        class_to_idx=self.class_to_idx

        # Filter classes to include only those in used_cluster
        # filtered_classes = [cls_name for cls_name in classes if cls_name in self.used_cluster]
        # filtered_classes=self.used_cluster
        # print(f"used_cluster while loading val={filtered_classes}")


        for cls_name in filtered_classes:
            cls_folder = os.path.join(val_dir, cls_name, 'images')
            image_files = os.listdir(cls_folder)

            for img_file in image_files:
                img_path = os.path.join(cls_folder, img_file)
                self.data.append(img_path)
                self.labels.append(class_to_idx[cls_name])

        # dataset = [(transform(Image.open(img_path).convert('RGB')), label) for img_path, label in zip(data, labels)]
        dataset = [self.data, self.labels]
        return dataset

    def _load_test_data(self):
        """
        Load test data for specified labels in used_cluster.

        Args:
            used_cluster (list): List of label names (classes) to load data for.
        """
        test_dir = os.path.join(self.data_dir, 'test')

        # Check if `used_cluster` is specified and not empty
        if self.used_cluster is not None and len(self.used_cluster) > 0:
            filtered_classes = self.used_cluster
        else:
            filtered_classes = sorted(os.listdir(test_dir))

        # classes = sorted(os.listdir(test_dir))
        # class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        class_to_idx=self.class_to_idx

        # Filter classes to include only those in used_cluster
        # filtered_classes = [cls_name for cls_name in classes if cls_name in self.used_cluster]
        # filtered_classes=self.used_cluster


        for cls_name in filtered_classes:
            cls_folder = os.path.join(test_dir, cls_name, 'images')
            image_files = os.listdir(cls_folder)

            for img_file in image_files:
                img_path = os.path.join(cls_folder, img_file)
                self.data.append(img_path)
                self.labels.append(class_to_idx[cls_name])

        # dataset = [(transform(Image.open(img_path).convert('RGB')), label) for img_path, label in zip(data, labels)]
        dataset = [self.data, self.labels]
        return dataset

    def __getitem__(self, index):
        img_path = self.data[index]
        label = self.labels[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    # def __getitem__(self, index, transform=None):
    #     """
    #     Fetch a single data point (image and label) by index.

    #     Args:
    #         index (int): Index of the data point.
    #         transform (callable, optional): Transform to apply to the image.

    #     Returns:
    #         img (torch.Tensor): Transformed image tensor.
    #         target (int): Corresponding label.
    #     """
    #     img_path = self.data[index]
    #     target = self.labels[index]

    #     # Load the image
    #     img = Image.open(img_path).convert('RGB')

    #     # Use the passed transform or fall back to the default transform
    #     if transform is not None:
    #         img = transform(img)
    #     elif self.default_transform is not None:
    #         img = self.default_transform(img)

    #     return img, target

    def __len__(self):
        return len(self.data)

    @staticmethod
    def create_dataset_group(original_data_dir, output_data_dir, used_clusters, group_order, all_num_clusters):
        """
        Create a grouped dataset based on the specified clusters.

        Args:
            original_data_dir (str): Path to the original TinyImageNet dataset.
            output_data_dir (str): Path to save the grouped dataset.
            used_clusters (list): List of cluster (class) indices to include in the group.
            group_order (int): Current group order.
            all_num_clusters (int): Total number of clusters.

        Returns:
            None
        """
        train_dir = os.path.join(original_data_dir, 'train')
        val_dir = os.path.join(original_data_dir, 'val')

        grouped_train_dir = os.path.join(output_data_dir, 'train')
        grouped_val_dir = os.path.join(output_data_dir, 'val')

        # Create output directories
        os.makedirs(grouped_train_dir, exist_ok=True)
        os.makedirs(grouped_val_dir, exist_ok=True)

        # Handle train data
        classes = sorted(os.listdir(train_dir))
        # if group_order == all_num_clusters:
        #     # Exclude classes with indices > 99
        #     filtered_classes = [cls_name for cls_name in classes if int(cls_name[1:]) <= 99]
        # else:
        #     # Include only the specified clusters
        #     filtered_classes = [classes[idx] for idx in used_clusters]

        for cls_name in classes:
            cls_folder = os.path.join(train_dir, cls_name, 'images')
            target_folder = os.path.join(grouped_train_dir, cls_name)
            os.makedirs(target_folder, exist_ok=True)        
            # List all image files in the class folder
            img_files = os.listdir(cls_folder)
            # Shuffle the image files
            random.shuffle(img_files)
            # Select only 250 samples
            selected_images = img_files[:250]
            # Copy the selected images to the target folder
            for img_file in selected_images:
                shutil.copy(os.path.join(cls_folder, img_file), os.path.join(target_folder, img_file))

        # Handle validation data
        for cls_name in classes:
            cls_folder = os.path.join(val_dir, cls_name, 'images')
            target_folder = os.path.join(grouped_val_dir, cls_name, 'images')
            os.makedirs(target_folder, exist_ok=True)

            for img_file in os.listdir(cls_folder):
                shutil.copy(os.path.join(cls_folder, img_file), os.path.join(target_folder, img_file))

    @staticmethod
    def convert_to_range(labels, class_to_idx):
        """
        Convert labels to a continuous range starting from 0.
        Args:
            labels (list): List of original labels.
            class_to_idx (dict): Mapping of class names to indices.
        Returns:
            list: Labels converted to a continuous range.
        """
        return [class_to_idx[label] for label in labels]


def cal_mean_std_group(data_dir, split='train', batch_size=64, transform=None):
    """
    Calculate the mean and standard deviation of the grouped dataset.

    Args:
        data_dir (str): Path to the grouped dataset directory.
        split (str): Dataset split - 'train' or 'val'.
        batch_size (int): Batch size for loading the dataset.

    Returns:
        tuple: Means and standard deviations for each channel.
    """
    # Define transformations
    # transform = transforms.Compose([
    #     transforms.Resize((64, 64)),
    #     transforms.ToTensor()
    # ])

    dataset = TinyImageNetDataset(data_dir, split=split, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize mean and std tensors
    mean = torch.zeros(3)  # For RGB channels
    std = torch.zeros(3)
    total_images_count = 0

    # Iterate over batches to calculate mean and std
    for images, _ in data_loader:
        batch_samples = images.size(0)  # Number of images in the current batch
        images = images.view(batch_samples, images.size(1), -1)  # Flatten height and width into one dimension
        mean += images.mean(2).sum(0)  # Sum over all pixels in the batch
        std += images.std(2).sum(0)  # Sum standard deviation over all pixels in the batch
        total_images_count += batch_samples

    # Avoid division by zero if dataset is empty
    if total_images_count == 0:
        raise ValueError("The dataset is empty. Check the data directory or split parameter.")

    # Normalize mean and std by the total number of images
    mean /= total_images_count
    std /= total_images_count

    # Convert mean and std to lists for better readability
    return mean.tolist(), std.tolist()