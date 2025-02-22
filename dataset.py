import os

import cv2
import torch
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import itertools

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
    def __init__(self, root_video_dir, root_mask_dir, transform=None):
          super().__init__()

          self.root_vid = root_video_dir
          self.root_mask = root_mask_dir

          self.class_colors = {
               0: (255, 0, 0),        # Grasper: Red
               1: (0, 255, 0),        # bipolar: Green
               2: (0, 0, 255),        # Hook: Blue
               3: (255, 255, 0),      # Clipper: Yellow
               4: (0, 255, 255),      # clipper: Cyan
               5: (255, 0, 255),      # Irrigator: Magenta
          }

          if transform == None:
               self.transforms = transforms.Compose([
                    # transforms.Resize((256,256)), # might have to change the value from 256 X 256 to 480 X 854
                    #   transforms.Resize((480, 854)),
                    transforms.ToTensor(),
               ])
          else:
               self.transforms = transform

          video_list = sorted(os.listdir(self.root_vid))
         
         # this gives a list of list for frames, [vid1/frames, vid2/frames, ...]
          self.frame_path = []
          self.mask_path = []
          
          for vid in video_list:
               video_path = os.path.join(root_video_dir, vid)
               mask_vid_path = os.path.join(root_mask_dir, vid)
               
               frame_path = sorted([os.path.join(video_path,frame) for frame in os.listdir(video_path)])
               mask_path = sorted([os.path.join(mask_vid_path, frame) for frame in os.listdir(mask_vid_path)])
               
               self.frame_path.append(frame_path)
               self.mask_path.append(mask_path)

          self.frame_path_sequenced = list(itertools.chain(*self.frame_path))
          self.mask_path_sequenced = list(itertools.chain(*self.mask_path))

    def _convert_to_segmentation_mask(self, mask):
        """
        Converts the RGB mask into a one-hot encoded format where each class
        gets its own channel, and each pixel is either 0 or 1 depending on the class.
        """
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((len(self.class_colors), height, width), dtype=np.float32)  # (C, H, W)

        for label_index, (class_id, color) in enumerate(self.class_colors.items()):
            segmentation_mask[label_index] = np.all(mask == color, axis=-1).astype(float)  # Binary mask per class

        return segmentation_mask  # Shape: (C, H, W)

    def __len__(self):
        return len(self.frame_path_sequenced)

    def __getitem__(self, index):
        # Load image
        image = cv2.imread(self.frame_path_sequenced[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask = cv2.imread(self.mask_path_sequenced[index], cv2.IMREAD_UNCHANGED)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  # Ensure it has 3 channels

        # Resize
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)  # Preserve visual quality
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)  # Preserve labels

        # Convert to segmentation mask
        mask = self._convert_to_segmentation_mask(mask=mask)  # (C, H, W)

        # Convert image to PIL and apply transforms
        image = Image.fromarray(image)
        image = self.transforms(image)

        return image, torch.from_numpy(mask)  # Ensure mask is a tensor

            
if __name__=='__main__':

    dataset = SurgiSeg(root_mask_dir='dataset/Masks', root_video_dir='dataset/videos_batched')
