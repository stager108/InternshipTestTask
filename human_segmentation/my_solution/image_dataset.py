import torch.utils.data as dt
import torch
import os
import random
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import rotate
from scipy.ndimage import zoom
from augmentation import *

to_img = ToPILImage()
      
last_preprocess = transforms.Compose([
    transforms.RandomGrayscale(),
    transforms.ToTensor()
])

vflip = transforms.Compose([
    transforms.RandomVerticalFlip(p=1.0)
])

hflip = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0)
])

colorjitter = transforms.Compose([
    transforms.ColorJitter()
])


class ImageDataset(dt.Dataset):
  
    def __init__(self, data_path, mask_path, augment=True ):
      
        self.files = os.listdir(data_path)
        self.files.sort()
        self.mask_files = os.listdir(mask_path)
        self.mask_files.sort()
        self.data_path = data_path
        self.mask_path = mask_path
        self.augment = augment 


    def __len__(self):
        return len(self.files)
      
      
    def simple_preprocess(self, input, target):
        # Preprocessing without any augmentation
        _preprocess = transforms.Compose([
            transforms.ToTensor()
        ])
        return _preprocess(input), _preprocess(target)
        
    def hard_preprocess(self, input, target):
        # Preprocessing with augmentation
        
        p = (np.random.uniform(size=15))
        
        if p[0]> 0.5:
            input, target = random_rotate(input, target, 90)
            
        #if p[1] < 0.4:
         #   input, target = random_crop(input, target, 50)
        if p[2] > 0.5:
            input, target = random_zoom(input, target, 20)
        
        if p[3] > 0.3:
            input = colorjitter(input)
            
        if p[4] > 0.5:
            input = hflip(input)     
            target = hflip(target)
           
        if p[5] > 0.7:
            input = vflip(input)     
            target = vflip(target)

     #   if p[6] > 0.8:
      #      input = color_shuffle(input)
            
       # if p[7] > 0.8:
        #    input = random_color(input)
            
    #    if p[8] > 0.8:
     #       input = ImageOps.invert(input)     
            
        if p[9] > 0.8:
            input = sp_noise(input, 0.02)
        elif p[9] < 0.1:
            input = sp_noise1(input, 0.02)
                              
        elif p[10] > 0.8:
            input = gaussian_noise(input, 5)
        
        if p[11]> 0.9:
            input = random_shift(input, 5)
            
        return last_preprocess(input), last_preprocess(target)
        
    def pil_load(self, path, is_input=True):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def pil_save(self, t, img_path):
        a = to_img(t)
        a.save(img_path, 'PNG')

    def __getitem__(self, idx):
        f_name = os.path.join(self.data_path, self.files[idx])
        m_name = os.path.join(self.mask_path, self.mask_files[idx])

        input = self.pil_load(f_name)
        target = self.pil_load(m_name, False)
        
        if self.augment == True:
            input, target = self.hard_preprocess(input, target)
        else:
            input, target = self.simple_preprocess(input, target)
        
        target = torch.sum(target, dim=0).unsqueeze(0)
        target[ torch.gt(target, 0) ] = 1

        return input, target

class OnlyImageDataset(dt.Dataset):
  
    def __init__(self, data_path):
      
        self.files = os.listdir(data_path)
        self.files.sort()
        self.data_path = data_path
        
    def __len__(self):
        return len(self.files)
      
    def simple_preprocess(self, input):
        # Preprocessing without any augmentation
        _preprocess = transforms.Compose([
            transforms.ToTensor()
        ])
        return _preprocess(input)
        
    def pil_load(self, path, is_input=True):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def pil_save(self, t, img_path):
        a = to_img(t)
        a.save(img_path, 'PNG')

    def __getitem__(self, idx):
        f_name = os.path.join(self.data_path, self.files[idx])

        input = self.pil_load(f_name)
        input= self.simple_preprocess(input)
    
        return input