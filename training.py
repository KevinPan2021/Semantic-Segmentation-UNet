import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np

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
def mean_iou(pred_mask, mask , n_classes=23):
    smooth = 1e-10 # prevent divide by zero
    
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect+smooth) / (union+smooth) 
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class) * 100
    
    

# Mean Pixel Accuracy
def mean_pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy * 100


# forward propagation
def feedforward(model, nclasses, data_loader, GPU_Device):
    epoch_loss = 0.0
    epoch_MIoU = 0.0
    epoch_acc = 0.0
    
    # Define loss function
    #criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss()
    
    # Set model to evaluation mode
    model.eval()
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        
        # Iterate over the dataset
        for X, Y in data_loader:
            # move to device
            X = X.to(GPU_Device)
            Y = Y.to(GPU_Device)
            
            # Forward pass
            output = model(X)
            
            # compute loss
            loss = criterion(output, Y)
            
            # compute predicted
            _, predicted = torch.max(output, 1)
            
            # Update the statistics
            epoch_loss += loss.item()
            epoch_MIoU += mean_iou(output, Y, n_classes=nclasses)
            epoch_acc += mean_pixel_accuracy(output, Y)

    # Calculate test accuracy and loss
    epoch_acc /= len(data_loader)
    epoch_loss /= len(data_loader)
    epoch_MIoU /= len(data_loader)
    return epoch_acc, epoch_MIoU, epoch_loss




def backpropagation(model, nclasses, data_loader, optimizer, scheduler, GPU_Device):
    epoch_loss = 0.0
    epoch_MIoU = 0.0
    epoch_acc = 0.0
    
    # Define loss function
    #criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss()
    
    # Set model to training mode
    model.train()
    
    # Iterate over the dataset
    for X, Y in tqdm(data_loader):
        
        # move to device
        X = X.to(GPU_Device)
        Y = Y.to(GPU_Device)
        
        # Forward pass
        output = model(X)
        
        # compute loss
        loss = criterion(output, Y)
        
        # compute predicted
        _, predicted = torch.max(output, 1)
        
        # Update test statistics
        epoch_loss += loss.item()
        epoch_MIoU += mean_iou(output, Y, n_classes=nclasses)
        epoch_acc += mean_pixel_accuracy(output, Y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
    
    # Update the learning rate
    scheduler.step()
    
    # Calculate test accuracy and loss
    epoch_acc /= len(data_loader)
    epoch_loss /= len(data_loader)
    epoch_MIoU /= len(data_loader)
    
    return epoch_acc, epoch_MIoU, epoch_loss

   

def model_training(model, nclasses, train_loader, valid_loader, GPU_Device):    
    # Define hyperparameters
    learning_rate = 0.002
    weight_decay = 0
    num_epochs = 50
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay, lr=learning_rate)
    
    # learning rate scheduler - gradually decrease learning rate over time
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # calculate the initial statistics (random)
    train_acc, train_MIoU, train_loss = feedforward(model, nclasses, train_loader, GPU_Device)
    valid_acc, valid_MIoU, valid_loss = feedforward(model, nclasses, valid_loader, GPU_Device)
    train_acc_curve, train_loss_curve, train_MIoU_curve = [train_acc], [train_loss], [train_MIoU]
    valid_acc_curve, valid_loss_curve, valid_MIoU_curve = [valid_acc], [valid_loss], [valid_MIoU]
    print(f"Epoch 0/{num_epochs}")
    print(f"Train Loss: {train_loss:.3f} | MIoU: {train_MIoU:.2f}% | Acc: {train_acc:.2f}%")
    print(f"Valid Loss: {valid_loss:.3f} | MIoU: {valid_MIoU:.2f}%| Acc: {valid_acc:.2f}%")
          
    # Early Stopping criteria
    patience = 3
    not_improved = 0
    best_valid_loss = valid_loss
    threshold = 0.01
    
    # Training loop
    for epoch in range(num_epochs):
        train_acc, train_MIoU, train_loss = backpropagation(model, nclasses, train_loader, optimizer, scheduler, GPU_Device)
        valid_acc, valid_MIoU, valid_loss = feedforward(model, nclasses, valid_loader, GPU_Device)
        
        train_acc_curve.append(train_acc)
        train_loss_curve.append(train_loss)
        train_MIoU_curve.append(train_MIoU)
        valid_acc_curve.append(valid_acc)
        valid_loss_curve.append(valid_loss)
        valid_MIoU_curve.append(valid_MIoU)
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.3f} | MIoU: {train_MIoU:.2f}% | Acc: {train_acc:.2f}%")
        print(f"Valid Loss: {valid_loss:.3f} | MIoU: {valid_MIoU:.2f}%| Acc: {valid_acc:.2f}%")
        
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
    plot_training_curves(train_acc_curve, train_MIoU_curve, train_loss_curve, \
                         valid_acc_curve, valid_MIoU_curve, valid_loss_curve)