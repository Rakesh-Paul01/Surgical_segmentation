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
    """

    Args:
        Dataset (torch.utils.data.Dataset): The dataset class of pytorch
    """
    def __init__(self, root_video_dir: str, 
                 root_mask_dir: str, 
                 transform: torchvision.transforms= None
                 ):
          

          super().__init__()

          self.root_vid = root_video_dir
          self.root_mask = root_mask_dir

          self.class_colors: Dict[int,Tuple[int, int, int]] = {
               0: (255, 0, 0),        # Grasper: Red
               1: (0, 255, 0),        # bipolar: Green
               2: (0, 0, 255),        # Hook: Blue
               3: (255, 255, 0),      # Clipper: Yellow
               4: (0, 255, 255),      # clipper: Cyan
               5: (255, 0, 255),      # Irrigator: Magenta
               6: (0,0,0),             # black: Background
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

          height, width = mask.shape[:2]
          segmentation_mask = np.zeros((height, width, len(self.class_colors)), dtype= np.float32)
          for idx, label in self.class_colors.items():
               segmentation_mask[:, :, int(idx)] = np.all(mask == label, axis= -1).astype(float)
               
          return segmentation_mask
          

    def __len__(self):
        return len(self.frame_path_sequenced)

    def __getitem__(self, index):
        # reading the image in pytorch
     #    image = read_image(self.frame_path_sequenced[index])
     #    image = self.transforms(img= image)

     #    # reading the mask in pytorch
     #    mask = read_image(self.mask_path_sequenced[index])
     #    mask = self.transforms(img=mask)

        mask = cv2.imread(self.mask_path_sequenced[index])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        image = cv2.imread(self.frame_path_sequenced[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        
        mask = self._convert_to_segmentation_mask(mask=mask)

     #    image, mask = self.transforms(image), self.transforms(mask)

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

if __name__=='__main__':

    dataset = SurgiSeg(root_mask_dir='dataset/Masks', root_video_dir='dataset/original')
