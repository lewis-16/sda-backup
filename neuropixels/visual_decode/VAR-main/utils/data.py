import os.path as osp

import PIL.Image as PImage
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms

import os
import os.path as osp
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class NPYImageDataset(Dataset):
    def __init__(self, npy_path, image_dir, transform=None):
        self.data = np.load(npy_path)
        self.image_dir = image_dir
        self.transform = transform
        
        assert self.data.shape[1] == 1025, f"数据形状应为(n, 1025)，但得到{self.data.shape}"
        
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.tiff', '.jpg', '.jpeg', '.bmp'))]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        latent_vector = self.data[idx, :1024] 
        image_label = int(self.data[idx, 1024]) 
        
        image_path = osp.join(self.image_dir, f"natural_scene_{image_label}.tiff")
        
        try:
            img = Image.open(image_path).convert('L') 
        except FileNotFoundError:
            img = Image.new('L', (256, 256), color=0)
            print(f"Warning: image {image_path} disappear")
        
        if self.transform:
            img = self.transform(img)
        
        latent_vector = torch.tensor(latent_vector, dtype=torch.float32)
        
        return img, latent_vector

def build_dataset(
    data_path: str, 
    final_reso: int
):
    transform = transforms.Compose([
        transforms.Resize((final_reso, final_reso)),  
        transforms.ToTensor(),                       
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Lambda(lambda x: x * 2 - 1)         
    ])
    
    train_set = NPYImageDataset(
        npy_path=osp.join(data_path, 'VISp_train_features.npy'),
        image_dir=osp.join('/media/ubuntu/sda/neuropixels/nature_scene'),
        transform=transform
    )
    
    val_set = NPYImageDataset(
        npy_path=osp.join(data_path, 'VISp_eval_features.npy'),
        image_dir=osp.join('/media/ubuntu/sda/neuropixels/nature_scene'),
        transform=transform
    )
    
    num_classes = 1000  
    
    return num_classes, train_set, val_set


# def build_dataset(
#     data_path: str, final_reso: int,
#     hflip=False, mid_reso=1.125,
# ):
#     # build augmentations
#     mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
#     train_aug, val_aug = [
#         transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
#         transforms.RandomCrop((final_reso, final_reso)),
#         transforms.ToTensor(), normalize_01_into_pm1,
#     ], [
#         transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
#         transforms.CenterCrop((final_reso, final_reso)),
#         transforms.ToTensor(), normalize_01_into_pm1,
#     ]
#     if hflip: train_aug.insert(0, transforms.RandomHorizontalFlip())
#     train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
    
#     # build dataset
#     train_set = DatasetFolder(root=osp.join(data_path, 'train'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=train_aug)
#     val_set = DatasetFolder(root=osp.join(data_path, 'val'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=val_aug)
#     num_classes = 1000
#     print(f'[Dataset] {len(train_set)=}, {len(val_set)=}, {num_classes=}')
#     print_aug(train_aug, '[train]')
#     print_aug(val_aug, '[val]')
    
#     return num_classes, train_set, val_set


def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')
