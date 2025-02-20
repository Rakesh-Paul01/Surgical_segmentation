import torch
from dataset import SurgiSeg
from torch.utils.data import DataLoader

'''
create the model and the training module
model can mostly be the existing ones
training module you have the check and verify for self
'''


from torchvision.models.segmentation import deeplabv3_resnet50

deeplabv3 = deeplabv3_resnet50(
    weights='COCO_WITH_VOC_LABELS_V1', 
    weights_backbone='IMAGENET1K_V1'
)

# change outputs to desired number of classes
# deeplabv3.classifier[4] = torch.nn.Conv2d(256, 6, kernel_size=(1, 1), stride=(1, 1))
# print(deeplabv3)

print(deeplabv3.classifier[4])