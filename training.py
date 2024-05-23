import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import GradScaler, autocast

from visualization import plot_training_curves



# use focal loss to mitigate data imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute Cross Entropy Loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Compute Probabilities and Logits
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma) * ce_loss

        # Apply Reduction
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss
        
       
# Mean Intersection over Union
@torch.no_grad
def mean_iou(pred_mask, mask, n_classes):
    smooth = 1e-10 # prevent divide by zero
    
    pred_mask = F.softmax(pred_mask, dim=1)
    pred_mask = torch.argmax(pred_mask, dim=1)
    pred_mask = pred_mask.contiguous().view(-1)
    mask = mask.contiguous().view(-1)

    iou_per_class = []
    for clas in range(0, n_classes): #loop per pixel class
        true_class = pred_mask == clas
        true_label = mask == clas
        
        # no label exist
        if true_label.long().sum().item() == 0: 
            iou_per_class.append(np.nan)
        else:
            intersect = torch.logical_and(true_class, true_label).sum().float().item()
            union = torch.logical_or(true_class, true_label).sum().float().item()

            iou = (intersect+smooth) / (union+smooth) 
            iou_per_class.append(iou)
    return np.nanmean(iou_per_class) * 100
    
    

# Mean Pixel Accuracy
@torch.no_grad
def mean_pixel_accuracy(output, mask):
    output = torch.argmax(F.softmax(output, dim=1), dim=1)
    correct = torch.eq(output, mask).int()
    accuracy = float(correct.sum()) / float(correct.numel()) * 100
    return accuracy


# forward propagation
@torch.no_grad
def feedforward(model, dataloader):
    # Set model to evaluation mode
    model.eval()
    
    running_acc = 0.0
    running_loss = 0.0
    running_MIoU = 0.0
    
    # Define loss function
    criterion = FocalLoss()
    
    device = next(model.parameters()).device
    nclasses = model.nclasses()
    
    # Iterate over the dataset
    with tqdm(total=len(dataloader)) as pbar:
        for i, (X, Y) in enumerate(dataloader):
            # move to device
            X = X.to(device)
            Y = Y.to(device)
            
            # mixed precision
            with autocast(dtype=torch.float16):
                output = model(X)
                loss = criterion(output, Y)
            
            # compute predicted
            _, predicted = torch.max(output, 1)
            
            # Update statistics
            running_loss += loss.item()
            running_MIoU += mean_iou(output, Y, nclasses)
            running_acc += mean_pixel_accuracy(output, Y)
            
            # Update tqdm description with loss, accuracy, and f1 score
            pbar.set_postfix({
                'Loss': running_loss/(i+1), 
                'Acc': round(running_acc/(i+1),1),
                'MIoU': round(running_MIoU/(i+1),1)
            })
            pbar.update(1)
            
            
    # Calculate test accuracy and loss
    running_acc /= len(dataloader)
    running_MIoU /= len(dataloader)
    running_loss /= len(dataloader)
    
    return running_acc, running_MIoU, running_loss



def backpropagation(model, dataloader, optimizer, scaler):
    # Set model to training mode
    model.train()
    
    running_acc = 0.0
    running_loss = 0.0
    running_MIoU = 0.0
    
    # Define loss function
    criterion = FocalLoss()
    
    device = next(model.parameters()).device
    nclasses = model.nclasses()
    
    # Iterate over the dataset
    with tqdm(total=len(dataloader)) as pbar:
        for i, (X, Y) in enumerate(dataloader):
            # move to device
            X = X.to(device)
            Y = Y.to(device)
            
            # mixed precision
            with autocast(dtype=torch.float16):
                output = model(X)    
                loss = criterion(output, Y)
            
            # compute predicted
            _, predicted = torch.max(output, 1)
            
            # Update statistics
            running_loss += loss.item()
            running_MIoU += mean_iou(output, Y, nclasses)
            running_acc += mean_pixel_accuracy(output, Y)
            
            # Reset gradients
            optimizer.zero_grad()
    
            # Backpropagate the loss
            scaler.scale(loss).backward()
    
            # Optimization step
            scaler.step(optimizer)
    
            # Updates the scale for next iteration.
            scaler.update()
            
            # Update tqdm description with loss, accuracy, and f1 score
            pbar.set_postfix({
                'Loss': running_loss/(i+1), 
                'Acc': round(running_acc/(i+1),1),
                'MIoU': round(running_MIoU/(i+1),1)
            })
            pbar.update(1)
            
    
    # Calculate test accuracy and loss
    running_acc /= len(dataloader)
    running_MIoU /= len(dataloader)
    running_loss /= len(dataloader)
    
    return running_acc, running_MIoU, running_loss

   

def model_training(model, train_loader, valid_loader):    
    # Define hyperparameters
    learning_rate = 5e-5
    n_epochs = 50
    
    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler()
    
    # calculate the initial statistics
    print(f"Epoch 0/{n_epochs}")
    train_acc, train_MIoU, train_loss = feedforward(model, train_loader)
    valid_acc, valid_MIoU, valid_loss = feedforward(model, valid_loader)
    train_accs, train_losses, train_MIoUs = [train_acc], [train_loss], [train_MIoU]
    valid_accs, valid_losses, valid_MIoUs = [valid_acc], [valid_loss], [valid_MIoU]
    
        
    # Early Stopping criteria
    patience = 3
    not_improved = 0
    best_valid_loss = valid_loss
    threshold = 0.01
    
    # Training loop
    for epoch in range(n_epochs):
        print(f'Epoch {epoch+1}/{n_epochs}')
        train_acc, train_MIoU, train_loss = backpropagation(model, train_loader, optimizer, scaler)
        valid_acc, valid_MIoU, valid_loss = feedforward(model, valid_loader)
        
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        train_MIoUs.append(train_MIoU)
        valid_accs.append(valid_acc)
        valid_losses.append(valid_loss)
        valid_MIoUs.append(valid_MIoU)
        
        # evaluate the current preformance
        if valid_loss < best_valid_loss + threshold:
            best_valid_loss = valid_loss
            not_improved = 0
            # save the best model based on validation loss
            torch.save(model.state_dict(), f'{type(model).__name__}.pth')
        else:
            not_improved += 1
            if not_improved >= patience:
                print('Early Stopping Activated')
                break
        
    # plotting the training curves
    plot_training_curves(
        train_accs, train_MIoUs, train_losses, 
        valid_accs, valid_MIoUs, valid_losses
    )