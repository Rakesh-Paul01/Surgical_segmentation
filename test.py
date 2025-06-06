import os
import torch
from torch import nn
import torch.nn.functional as F
from models import Unet
from torch.utils.data import DataLoader
from dataset import SurgiSeg
import matplotlib.pyplot as plt
import numpy as np


# model = Unet(3, 7)
# print(torch.randn((3, 3)))

def visualize(img):
    img_np = img.permute(1, 2, 0).numpy()

    # If the image values are in [0, 1] range, no need to normalize
    # If they're in [0, 255] range, you might need to divide by 255
    # Let's check the range first
    print(f"Image value range: {img_np.min()} to {img_np.max()}")

    # Display the image
    plt.figure(figsize=(10, 5))

    # Show the original image
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title('Original Image')
    plt.axis('off')

def view_mask(mask):
    fig, axes = plt.subplots(1, 7, figsize=(20, 5))
    for i in range(7):
        axes[i].imshow(mask[:, :, i], cmap='gray')
        axes[i].set_title(f'Class {i}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ =='__main__':
    
    
    trainset = SurgiSeg(root_video_dir='data/original', root_mask_dir='data/Masks')
    testset =  SurgiSeg(root_video_dir='data/test/original', root_mask_dir='data/test/Masks')

    img, mask = trainset[23980]
    print('this is the shape of the image', img.shape)
    print('this is the shape of the mask', mask.shape)
    
    # for i in range(mask.shape[2]):
    #     print((mask[:,:,0] == mask[:,:, i]).all())
    #     print(mask[:,:,i].shape)

    # visualize(img=img)
    # view_mask(mask=mask)