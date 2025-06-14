import os
import itertools
from typing import Dict, Tuple, List

import cv2
import torch
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision
# from torchvision import transforms

'''
TODO:
    For the init we have a path to the mask and a path to the original image based on the sequencing index, stored in the 
    self.frame_paths , self.mask_paths
    
    especially the for loop, frame_path, mask_path
    using the same check the __len__ and implement the __getitem__
'''
'''
the below part should also include the presence of mutiple mask and  video directories and
each of them should be added to the frame_paths and mask_paths. 


'''

class SurgiSeg(Dataset):
    """ The torch.utils.data.Dataset for the SurgiSeg work. This Dataclass can be used to get the mask and the images 
    for training segmentation models using the SurgiSeg dataset
    """
    def __init__(self, root_video_dir: str, 
                 root_mask_dir: str, 
                 transform: torchvision.transforms= None
                 ):
          

          super().__init__()

          self.root_vid = root_video_dir
          self.root_mask = root_mask_dir

          self.class_colors: Dict[int,Tuple[int, int, int]] = {
               0: (0, 0, 0),             # background: black
               1: (255, 0, 0),        # Grasper: Red
               2: (0, 255, 0),        # bipolar: Green
               3: (0, 0, 255),        # Hook: Blue
               4: (255, 255, 0),      # Clipper: Yellow
               5: (0, 255, 255),      # clipper: Cyan
               6: (255, 0, 255),      # Irrigator: Magenta
               # 6: (0,0,0),             # black: Background
          }

          if transform == None:
               self.transforms = torchvision.transforms.Compose([
                    # torchvision.transforms.ToPILImage(),
                    # torchvision.transforms.Resize((256,256)),
                    torchvision.transforms.ToTensor(),
               ])
          else:
               self.transforms = transform

          video_list: List[str] = sorted(os.listdir(self.root_vid))
         
         # this gives a list of list for frames, [vid1/frames, vid2/frames, ...]
          self.frame_path: List[List[str]] = [] # this if for the video basis
          self.mask_path: List[List[str]] = []
          
          for vid in video_list:
               video_path = os.path.join(root_video_dir, vid)
               mask_vid_path = os.path.join(root_mask_dir, vid)
               
               frame_path = sorted([os.path.join(video_path,frame) for frame in os.listdir(video_path)])
               mask_path = sorted([os.path.join(mask_vid_path, frame) for frame in os.listdir(mask_vid_path)])
               
               self.frame_path.append(frame_path)
               self.mask_path.append(mask_path)

          self.frame_path_sequenced = list(itertools.chain(*self.frame_path)) # This is when we apply it on image basis
          self.mask_path_sequenced = list(itertools.chain(*self.mask_path))

    def _convert_to_segmentation_mask(self, mask):
          """Generates the mask with the shape (height, width, channels) where the channels is of the
          same dimension as that of the output number of classes

          Args:
               mask (np.ndarray): mask as read using cv2

          Returns:
               np.ndarray: return the mask with multiple channels, where each channel corresponds to 
               each class
          """
          height, width = mask.shape[:2]
          segmentation_mask = np.zeros((height, width, len(self.class_colors)), dtype= np.float32)
          for idx, label in self.class_colors.items():
               segmentation_mask[:, :, int(idx)] = np.all(mask == label, axis= -1).astype(float)
               
          return segmentation_mask
    
    def _convert_to_gray_scale_mask(self, mask):
          """Generates the mask with the shape (height, width)

          Args:
               mask (np.ndarray): The mask as read using cv2 this is a gray scale image

          Returns:
               np.ndarray: Return the mask with the shape of (height, width)
          """
          height, width = mask.shape[:2]
          segmentation_mask = np.zeros((height, width), dtype=np.float32)

          for idx, color in self.class_colors.items():
               color_match = np.all(mask == color, axis=2)
               segmentation_mask[color_match] = idx 

          return segmentation_mask



    def __len__(self):
        return len(self.frame_path_sequenced)

    def __getitem__(self, index):

        mask = cv2.imread(self.mask_path_sequenced[index])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        image = cv2.imread(self.frame_path_sequenced[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)  # Preserve visual quality
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)  # Preserve labels

        mask = self._convert_to_segmentation_mask(mask=mask)
        image = self.transforms(img=image)
        mask = self.transforms(img=mask)

        return image, mask


        # Load mask
        mask = cv2.imread(self.mask_path_sequenced[index], cv2.IMREAD_UNCHANGED)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  # Ensure it has 3 channels

        # Resize
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)  # Preserve visual quality
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)  # Preserve labels

        # Convert to segmentation mask
        mask = self._convert_to_segmentation_mask(mask=mask)  # (C, H, W)
        
     #    return mask

        # Convert image to PIL and apply transforms
        image = Image.fromarray(image)
        image = self.transforms(image)

        return image, torch.from_numpy(mask)  # Ensure mask is a tensor

            
def get_dataloader(dataset: torch.utils.data.Dataset, batch_size: int=128, shuffle: bool = True):
     return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                       num_workers=12, pin_memory=True)


def get_training_augmentation():
    import albumentations as A     
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0
        ),
        A.PadIfNeeded(min_height=320, min_width=320, always_apply=True),
        A.RandomCrop(height=320, width=320, always_apply=True),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return A.Compose(train_transform)

if __name__=='__main__':
    import matplotlib.pyplot as plt

    dataset = SurgiSeg(root_mask_dir='data/Masks', root_video_dir='data/original')

    image, mask = dataset[23980]
    print(f'the shape of the image is {image.shape}')