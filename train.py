import os
import shutil
import logging
from datetime import datetime
import logging

import numpy as np
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.utils.data import DataLoader

# def train(n_epochs, criterion, optimizer, trainset, testset, model, device=None):
#     if not device:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#     train_loader = DataLoader(dataset=trainset, batch_size=128, shuffle=True)
#     test_laoder = DataLoader(dataset=testset, batch_size=128, shuffle=True)

#     criterion = criterion
#     optimizer = optimizer

#     for epoch in range(n_epochs):
#         model.train()
#         epoch_loss = 0

#         for images, mask in train_loader:
#             images, mask = images.to(device), mask.to(device)
            
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, mask)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()

#     # model.eval()
#     # with torch.no_grad():

def evaluate(model, testset, device, batch_size=128, criterion=torch.nn.CrossEntropyLoss()):
    model.eval()
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
    
    total_iou = 0.0
    total_dice = 0.0
    total_loss = 0.0
    total_pixels = 0
    correct_pixels = 0
    num_classes = 7  # Ensure your dataset has this attribute
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            # Convert outputs to predicted class labels
            preds = torch.argmax(outputs, dim=1)
            
            # IoU and Dice Score calculation
            iou = compute_iou(preds, masks, num_classes)
            dice = compute_dice(preds, masks, num_classes)
            total_iou += iou
            total_dice += dice
            
            # Pixel accuracy
            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()
    
    avg_loss = total_loss / len(test_loader)
    avg_iou = total_iou / len(test_loader)
    avg_dice = total_dice / len(test_loader)
    pixel_accuracy = correct_pixels / total_pixels
    
    print(f"Test Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, Dice Score: {avg_dice:.4f}, Pixel Acc: {pixel_accuracy:.4f}")
    return {
        'loss': avg_loss,
        'iou': avg_iou,
        'dice': avg_dice,
        'pixel_acc': pixel_accuracy
    }

def compute_iou(preds, masks, num_classes):
    iou_scores = []
    for cls in range(num_classes):
        intersection = ((preds == cls) & (masks == cls)).sum().item()
        union = ((preds == cls) | (masks == cls)).sum().item()
        if union == 0:
            iou_scores.append(1.0)  # Ignore background if not present
        else:
            iou_scores.append(intersection / union)
    return np.mean(iou_scores)

def compute_dice(preds, masks, num_classes):
    dice_scores = []
    for cls in range(num_classes):
        intersection = ((preds == cls) & (masks == cls)).sum().item()
        total = (preds == cls).sum().item() + (masks == cls).sum().item()
        if total == 0:
            dice_scores.append(1.0)
        else:
            dice_scores.append(2 * intersection / total)
    return np.mean(dice_scores)


def train(trainset, testset, model, 
            criterion= torch.nn.CrossEntropyLoss, optimizer= torch.optim.Adam,
            batch_size=128, n_epochs= 50, device=None, model_name='deeplabv3'):
    
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f'using device: {device}')
        
    model.to(device)
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
    
    # Create log and weight directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    
    # Unique identifier for this training session
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"logs/{model_name}_bs{batch_size}_ep{n_epochs}_{timestamp}.log"
    weight_filename = f"weights/{model_name}_bs{batch_size}_ep{n_epochs}_{timestamp}.pth"
    best_weight_filename = f"weights/{model_name}_best.pth"

    # try:
    #     criterion_name = criterion.__name__
    # except:
    #     criterion_name = 'Dice_Loss'
    
    # Configure logging
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Training Details: Model: {model_name}, Optimizer: {optimizer.__name__}, Loss Function: {criterion_name}, Batch Size: {batch_size}, Epochs: {n_epochs}")

    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        log_message = f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}"
        print(log_message)
        logging.info(log_message)

        if (epoch + 1) % 5 == 0:
            eval_metrics = evaluate(model, testset, device, batch_size, criterion)
            if eval_metrics['loss'] < best_loss:
                best_loss = eval_metrics['loss']
                torch.save(model.state_dict(), best_weight_filename)
                logging.info(f"Best model saved at epoch {epoch+1} with loss {best_loss:.4f}")
                print(f"Best model saved at epoch {epoch+1} with loss {best_loss:.4f}")

    
    # Save model weights
    torch.save(model.state_dict(), weight_filename)
    logging.info(f"Model weights saved to {weight_filename}")
    print(f"Model weights saved to {weight_filename}")


    
            
    



    

