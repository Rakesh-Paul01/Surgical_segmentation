import os
import glob

import cv2
import torch
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

'''
TODO:
    For the init we have a path to the mask and a path to the original image based on the sequencing index, stored in the 
    self.frame_paths , self.mask_paths
    
    especially the for loop, frame_path, mask_path
    using the same check the __len__ and implement the __getitem__
'''

class SurgiSeg(Dataset):
    def __init__(self, mask_video_dir_path, video_path, transform=None):
        super().__init__()
        self.video_path = video_path
        self.mask_video_dir = mask_video_dir_path
        self.transforms = transform

        if transform == None:
             self.transforms = transforms.Compose([
                  transforms.Resize((256,256)),
                  transforms.ToTensor(),
             ])
        else:
             self.transforms = transform

        self.video_dirs = sorted(os.listdir(self.video_path))
        self.frame_paths = []
        self.mask_paths = []
        
        for video in self.video_dirs:
            base_name = video.split('.')[0]
            self.frame_paths.append(os.path.join(video_path, base_name + '.jpg'))
            self.mask_paths.append(os.path.join(mask_video_dir_path, base_name + '.png'))

    def __len__(self):
        return len(self.frame_paths)
    
    def __getitem__(self, index):
           img_path = self.frame_paths[index] 
           mask_path = self.mask_paths[index] 
           
           image = Image.open(img_path)
           mask = Image.open(mask_path)

           if self.transforms:
                image = self.transforms(image)
                mask = self.transforms(mask)

           return image, mask
             
            
if __name__=='__main__':
    dataset = SurgiSeg(mask_video_dir_path='dataset/mask/VID26/part1', video_path='dataset/VID26/part1')
    image, mask = dataset[0]

    print(image)
    print(mask)