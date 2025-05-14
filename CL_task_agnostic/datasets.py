import os
import numpy as np
import random
import torchio as tio
import torch
from torch.utils.data import Dataset, DataLoader

def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CustomDataset(Dataset):
    def __init__(self, subjects, transforms=None, transform_args=None):
        self.subjects = subjects
        self.transforms = transforms
        self.transform_args = transform_args

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        if self.transforms:
            subject = self.transforms(subject, self.transform_args)
        return subject

def train_transforms(subject, args):
    return tio.Compose([
        tio.ZNormalization(),
        # tio.RandomFlip(axes=(0,), flip_probability=0.5),
        # tio.RandomFlip(axes=(1,), flip_probability=0.5),
        tio.RandomElasticDeformation(
            num_control_points=tuple(args.num_control_points),
            max_displacement=tuple(args.max_displacement),
            locked_borders=1,
            image_interpolation='linear',
            label_interpolation='nearest'
        ),
        tio.OneHot(num_classes=2)
    ])(subject)

def test_transforms(subject):
    return tio.Compose([
        tio.ZNormalization(),
        tio.OneHot(num_classes=2)
    ])(subject)

def load_train_dataset_from_task(data_path, task_id, args):
    """
    Load train dataset from a specific task.
    
    Args:
        data_path (str): Path to the base dataset directory
        task_id (int): ID of the task to load train data from
        
    Returns:
        CustomDataset: Dataset containing train samples with transforms applied
    """
    # Train dataset from imagesTr and labelsTr
    images_tr_path = os.path.join(data_path, f'Task{task_id}', 'imagesTr')
    labels_tr_path = os.path.join(data_path, f'Task{task_id}', 'labelsTr')
    
    image_files_tr = sorted(os.listdir(images_tr_path))
    label_files_tr = sorted(os.listdir(labels_tr_path))
    # print('image_files_tr len',len(image_files_tr))
    
    base_subjects_tr = []
    for image_file,label_file in zip(image_files_tr,label_files_tr):
        mri = tio.ScalarImage(os.path.join(images_tr_path, image_file))
        segmentation=tio.LabelMap(os.path.join(labels_tr_path, label_file))
        if not np.allclose(mri.affine,segmentation.affine):
            image_affine = mri.affine
            new_label_affine = image_affine.copy()
            new_label = tio.LabelMap(tensor=segmentation.tensor, affine=new_label_affine)
            subject = tio.Subject(
                mri=mri,
                segmentation=new_label
                )
        else:
             subject = tio.Subject(
                 mri=mri,
                 segmentation=segmentation)
                      
        base_subjects_tr.append(subject)
    # print('base_subjects_tr size',len(base_subjects_tr))
    K = args.batch_size * args.batch_number
    train_dataset = CustomDataset(
        random.choices(base_subjects_tr, k=K), 
        transforms=train_transforms,
        transform_args=args
    )

    # train_dataset = CustomDataset(base_subjects_tr, transforms=train_transforms)
    
    return train_dataset

def load_test_dataset_from_task(data_path, task_id):
    """
    Load test dataset from a specific task.
    
    Args:
        data_path (str): Path to the base dataset directory
        task_id (int): ID of the task to load test data from
        
    Returns:
        CustomDataset: Dataset containing test samples with transforms applied
    """
    # Test dataset from imagesTs and labelsTs
    images_ts_path = os.path.join(data_path, f'Task{task_id}', 'imagesTs')
    labels_ts_path = os.path.join(data_path, f'Task{task_id}', 'labelsTs')
    
    image_files_ts = sorted(os.listdir(images_ts_path))
    label_files_ts = sorted(os.listdir(labels_ts_path))
    # print('image_files_ts len',len(image_files_ts))
    
    subjects_ts = []
    for image_file in image_files_ts:
        label_file = image_file
        mri = tio.ScalarImage(os.path.join(images_ts_path, image_file))
        segmentation = tio.LabelMap(os.path.join(labels_ts_path, label_file))
        
        # Fix affine if needed
        if not np.allclose(mri.affine, segmentation.affine):
            image_affine = mri.affine
            new_label_affine = image_affine.copy()
            new_label = tio.LabelMap(tensor=segmentation.tensor, affine=new_label_affine)
            subject = tio.Subject(mri=mri, segmentation=new_label)
        else:
            subject = tio.Subject(mri=mri, segmentation=segmentation)
            
        subjects_ts.append(subject)
    # print('subjects_ts size',len(subjects_ts))
        
    return CustomDataset(subjects_ts, transforms=test_transforms, transform_args=None)



def split_hippocampus_datasets(args):
    split_datasets = list()
    data_path = './datasets'
    # task_ids = args.task_ids
    for i, task_id in enumerate(args.task_ids):
        # Train and test dataset
        train_dataset = load_train_dataset_from_task(data_path, task_id,args)
        test_dataset = load_test_dataset_from_task(data_path, task_id)
        split_datasets.append([train_dataset, test_dataset])

    return split_datasets

# splited_dataset = split_hippocampus_datasets()

def build_continual_dataloader(splited_dataset,args):
    dataloader = list()
    for i, _ in enumerate(args.task_ids):
        dataset_train, dataset_val = splited_dataset[i]
        print('train_dataset size', len(dataset_train))
        print('test_dataset size', len(dataset_val))
        data_loader_train = DataLoader(
                dataset_train,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
        )
        data_loader_val = DataLoader(
                dataset_val,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        dataloader.append({'train': data_loader_train, 'val': data_loader_val})

    return dataloader

# data_loader = build_continual_dataloader(splited_dataset)

import matplotlib.pyplot as plt
import numpy as np

def visualize_middle_slice_with_overlay(sample,sample_unaugmented):
    """
    Visualize the middle axial slice (64x64) of a 3D image and its segmentation overlay.
    
    Args:
        sample: A dictionary containing 'mri' and 'segmentation' tensors
    """
    image = sample['mri'].data
    mask = sample['segmentation'].data

    image_unaugmented = sample_unaugmented['mri'].data
    mask_unaugmented = sample_unaugmented['segmentation'].data
    
    # Get the middle slice index along the depth dimension
    d_mid = image.shape[3] // 2  # 48 // 2 = 24
    
    # Extract the middle slice
    image_slice = image[0, :, :, d_mid].numpy()  # Shape: [64, 64]
    mask_slice = mask[1, :, :, d_mid].numpy()    # Shape: [64, 64], channel 1 (foreground)

    image_slice_unaugmented = image_unaugmented[0, :, :, d_mid].numpy()  # Shape: [64, 64]
    mask_slice_unaugmented = mask_unaugmented[1, :, :, d_mid].numpy()    # Shape: [64, 64], channel 1 (foreground)
    
    # Create a figure with 1 row and 3 columns
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    # Plot the augmented image slice
    im1 = axes[0].imshow(image_slice, cmap='gray')
    axes[0].set_title(f'Augmented MRI (z={d_mid})')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot the segmentation mask
    im2 = axes[1].imshow(mask_slice, cmap='hot')
    axes[1].set_title(f'Augmented Segmentation Mask (z={d_mid})')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Create overlay view
    im1 = axes[2].imshow(image_slice_unaugmented, cmap='gray')
    axes[2].set_title(f'Original MRI (z={d_mid})')
    axes[2].axis('off')
    plt.colorbar(im1, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Plot the segmentation mask
    im2 = axes[3].imshow(mask_slice_unaugmented, cmap='hot')
    axes[3].set_title(f'Original Segmentation Mask (z={d_mid})')
    axes[3].axis('off')
    plt.colorbar(im2, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

def visualize_first_n_samples(dataset, n=5):
    """
    Visualize the middle axial slices of the first n samples in a dataset.
    
    Args:
        dataset: Dataset containing samples with 'mri' and 'segmentation' fields
        n: Number of samples to visualize (default: 5)
    """
    # Limit to available samples
    n = min(n, len(dataset))
    
    # Create a figure with n rows and 3 columns
    fig, axes = plt.subplots(n, 4, figsize=(15, 5*n))
    
    for i in range(n):
        sample = dataset[0][i]
        sample_unaugmented = dataset[1][i]
        
        # Get the tensor data from the sample
        if hasattr(sample['mri'], 'tensor'):
            image_tensor = sample['mri'].tensor
        elif hasattr(sample['mri'], 'data'):
            image_tensor = sample['mri'].data
        else:
            image_tensor = sample['mri']
        
        if hasattr(sample['segmentation'], 'tensor'):
            mask_tensor = sample['segmentation'].tensor
        elif hasattr(sample['segmentation'], 'data'):
            mask_tensor = sample['segmentation'].data
        else:
            mask_tensor = sample['segmentation']

        if hasattr(sample_unaugmented['mri'], 'tensor'):
            image_tensor_unaugmented = sample_unaugmented['mri'].tensor
        elif hasattr(sample_unaugmented['mri'], 'data'):
            image_tensor_unaugmented = sample_unaugmented['mri'].data
        else:
            image_tensor_unaugmented = sample_unaugmented['mri']
        if hasattr(sample_unaugmented['segmentation'], 'tensor'):
            mask_tensor_unaugmented = sample_unaugmented['segmentation'].tensor
        elif hasattr(sample_unaugmented['segmentation'], 'data'):
            mask_tensor_unaugmented = sample_unaugmented['segmentation'].data
        else:
            mask_tensor_unaugmented = sample_unaugmented['segmentation']

        
        # Get the middle slice
        d_mid = image_tensor.shape[3] // 2  # 48 // 2 = 24
        
        # Extract the middle slices
        image_slice = image_tensor[0, :, :, d_mid].cpu().numpy()
        mask_slice = mask_tensor[1, :, :, d_mid].cpu().numpy()

        image_slice_unaugmented = image_tensor_unaugmented[0, :, :, d_mid].cpu().numpy()
        mask_slice_unaugmented = mask_tensor_unaugmented[1, :, :, d_mid].cpu().numpy()

        
        # For a single sample case
        if n == 1:
            ax_img, ax_mask, ax_img2, ax_mask2 = axes
        else:
            ax_img, ax_mask, ax_img2, ax_mask2 = axes[i]
        
        # Display original image
        im1 = ax_img.imshow(image_slice, cmap='gray')
        ax_img.set_title(f'Sample {i+1}: Augmented MRI')
        ax_img.axis('off')
        plt.colorbar(im1, ax=ax_img, fraction=0.046, pad=0.04)
        
        # Display segmentation mask
        im2 = ax_mask.imshow(mask_slice, cmap='hot')
        ax_mask.set_title(f'Sample {i+1}: Augmented Mask')
        ax_mask.axis('off')
        plt.colorbar(im2, ax=ax_mask, fraction=0.046, pad=0.04)
        
        im3 = ax_img2.imshow(image_slice_unaugmented, cmap='gray')
        ax_img2.set_title(f'Sample {i+1}: Original MRI')
        ax_img2.axis('off')
        plt.colorbar(im3, ax=ax_img2, fraction=0.046, pad=0.04)
        # Display segmentation mask
        im4 = ax_mask2.imshow(mask_slice_unaugmented, cmap='hot')
        ax_mask2.set_title(f'Sample {i+1}: Original Mask')
        ax_mask2.axis('off')
        plt.colorbar(im4, ax=ax_mask2, fraction=0.046, pad=0.04)


    
    plt.tight_layout()
    plt.show()