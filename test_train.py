from typing import Tuple
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from dataset import SurgiSeg, get_dataloader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from tqdm import tqdm
import os
from datetime import datetime
import numpy as np

def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, writer):
    """Train for one epoch and log metrics"""
    model.train()
    running_loss = 0.0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f'Training Epoch {epoch}')
    
    for batch_idx, (images, masks) in enumerate(progress_bar):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update running loss
        running_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Log batch-level metrics to TensorBoard
        global_step = epoch * num_batches + batch_idx
        writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)
    
    # Calculate average loss for the epoch
    avg_loss = running_loss / num_batches
    writer.add_scalar('Train/Epoch_Loss', avg_loss, epoch)
    
    return avg_loss

def validate_epoch(model, dataloader, loss_fn, device, epoch, writer):
    """Validate for one epoch and log metrics"""
    model.eval()
    running_loss = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f'Validation Epoch {epoch}')
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            
            running_loss += loss.item()
            progress_bar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
    
    # Calculate average validation loss
    avg_loss = running_loss / num_batches
    writer.add_scalar('Validation/Epoch_Loss', avg_loss, epoch)
    
    return avg_loss

def log_model_parameters(model, writer, epoch):
    """Log model parameters and gradients"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f'Parameters/{name}', param.data, epoch)
            writer.add_histogram(f'Gradients/{name}', param.grad.data, epoch)

def log_sample_predictions(model, dataloader, device, writer, epoch, num_samples=4):
    """Log sample predictions to TensorBoard"""
    model.eval()
    
    with torch.no_grad():
        # Get a batch of data
        images, masks = next(iter(dataloader))
        images = images[:num_samples].to(device)
        masks = masks[:num_samples].to(device)
        
        # Get predictions
        outputs = model(images)
        predictions = torch.sigmoid(outputs)  # Apply sigmoid for visualization
        
        # Convert to numpy for visualization
        images_np = images.cpu().numpy()
        masks_np = masks.cpu().numpy()
        predictions_np = predictions.cpu().numpy()
        
        # Log images, masks, and predictions
        for i in range(num_samples):
            # Original image
            writer.add_image(f'Sample_{i}/Image', images_np[i], epoch)
            
            # Ground truth mask (sum across classes for visualization)
            mask_viz = np.sum(masks_np[i], axis=0, keepdims=True)
            writer.add_image(f'Sample_{i}/Ground_Truth', mask_viz, epoch)
            
            # Prediction (sum across classes for visualization)
            pred_viz = np.sum(predictions_np[i], axis=0, keepdims=True)
            writer.add_image(f'Sample_{i}/Prediction', pred_viz, epoch)

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)

if __name__ == "__main__":
    # Configuration
    NUM_EPOCHS = 100
    SAVE_EVERY = 10  # Save checkpoint every N epochs
    LOG_EVERY = 5    # Log sample predictions every N epochs
    
    # Device setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset and dataloaders
    trainset = SurgiSeg(root_video_dir='data/original', root_mask_dir='data/Masks')
    testset = SurgiSeg(root_video_dir='data/test/original', root_mask_dir='data/test/Masks')
    trainloader = get_dataloader(dataset=trainset)
    testloader = get_dataloader(dataset=testset)
    
    print(f"Training samples: {len(trainset)}")
    print(f"Test samples: {len(testset)}")
    
    # Model setup
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=7,
    ).to(device)
    
    # Optimizer and loss function
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    loss_fn = DiceLoss(mode='multilabel', classes=list(range(7)), from_logits=True)
    
    # TensorBoard setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'runs/surgiseg_experiment_{timestamp}'
    writer = SummaryWriter(log_dir)
    
    # Create checkpoints directory
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"To view logs, run: tensorboard --logdir={log_dir}")
    
    # Log model architecture
    dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Adjust size as needed
    writer.add_graph(model, dummy_input)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-" * 50)
        
        # Training phase
        train_loss = train_epoch(model, trainloader, optimizer, loss_fn, device, epoch, writer)
        
        # Validation phase
        val_loss = validate_epoch(model, testloader, loss_fn, device, epoch, writer)
        
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Log model parameters and gradients
        log_model_parameters(model, writer, epoch)
        
        # Log sample predictions periodically
        if epoch % LOG_EVERY == 0:
            log_sample_predictions(model, testloader, device, writer, epoch)
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
            print(f"New best model saved with validation loss: {val_loss:.4f}")
        
        # Save periodic checkpoints
        if epoch % SAVE_EVERY == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
        
        # Log epoch summary
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Log epoch losses side by side for comparison
        writer.add_scalars('Loss_Comparison', {
            'Train': train_loss,
            'Validation': val_loss
        }, epoch)
    
    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, NUM_EPOCHS, val_loss, final_checkpoint_path)
    
    # Close TensorBoard writer
    writer.close()
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final model saved to: {final_checkpoint_path}")
    print(f"TensorBoard logs saved to: {log_dir}")
    print(f"To view logs, run: tensorboard --logdir={log_dir}")