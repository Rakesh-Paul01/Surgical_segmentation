import os
import glob

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

# class SurgiSeg(Dataset):
#     """Dataset class for training and create DataLoaders

#     Args:
#         Dataset (torch.utils.data): pytorch dataset class
#     """
#     def __init__(self, mask_video_dir_path, video_path, transform=None):
#         super().__init__()
#         self.video_path = video_path
#         self.mask_video_dir = mask_video_dir_path
#         self.transforms = transform

#         if transform == None:
#              self.transforms = transforms.Compose([
#                   transforms.Resize((256,256)), # might have to change the value from 256 X 256 to 480 X 854
#                 #   transforms.Resize((480, 854)),
#                   transforms.ToTensor(),
#              ])
#         else:
#              self.transforms = transform

#         self.video_dirs = sorted(os.listdir(self.video_path))
#         self.frame_paths = []
#         self.mask_paths = []
        
#         for video in self.video_dirs:
#             base_name = video.split('.')[0]
#             self.frame_paths.append(os.path.join(video_path, base_name + '.jpg'))
#             self.mask_paths.append(os.path.join(mask_video_dir_path, base_name + '.png'))

#     def __len__(self):
#         return len(self.frame_paths)
    
#     def __getitem__(self, index):
#            img_path = self.frame_paths[index] 
#            mask_path = self.mask_paths[index] 
           
#            image = Image.open(img_path)
#            mask = Image.open(mask_path)

#            if self.transforms:
#                 image = self.transforms(image)
#                 mask = self.transforms(mask)

#            return image, mask
       

class SurgiSeg(Dataset):
    def __init__(self, root_video_dir, root_mask_dir, transform=None):
          super().__init__()

          self.root_vid = root_video_dir
          self.root_mask = root_mask_dir

          if transform == None:
               self.transforms = transforms.Compose([
                    transforms.Resize((256,256)), # might have to change the value from 256 X 256 to 480 X 854
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

    def __len__(self):
         return len(self.frame_path_sequenced) # this is for all the frames in all the videos incase we want all the videos as a separate one then comment out the first line and uncomment the below one
     #     return len(self.frame_path) 

    def __getitem__(self, index):

          image = self.transforms(Image.open(self.frame_path_sequenced[index]))
          mask = self.transforms(Image.open(self.mask_path_sequenced[index]))
          
          return image, mask

            
if __name__=='__main__':
    import matplotlib.pyplot as plt
    dataset = SurgiSeg(root_mask_dir='dataset/Masks', root_video_dir='dataset/videos_batched')
     # dataset[0]

    image, mask = dataset[0]

    # Convert Tensor back to numpy for plotting (optional: use .cpu().numpy() if on CUDA)
    image = image.permute(1, 2, 0).numpy()  # Changing the shape to HxWxC for displaying
    mask = mask.permute(1, 2, 0).numpy()  # Same transformation for the mask
    
    # Plotting the image and mask side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)  # Display the image
    axes[0].set_title("Image")
    axes[0].axis('off')  # Hide axis for better visual appeal

    axes[1].imshow(mask, cmap='gray')  # Display the mask (assuming it's a binary mask)
    axes[1].set_title("Mask")
    axes[1].axis('off')  # Hide axis

    plt.show()