from torchvision.models.segmentation import deeplabv3_resnet50

deeplabv3 = deeplabv3_resnet50(
    weights='COCO_WITH_VOC_LABELS_V1',
    weights_backbone='IMAGENET1K_V1' 
)