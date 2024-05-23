import matplotlib.pyplot as plt
import numpy as np


# plot the loss and acc curves
def plot_training_curves(train_acc, train_MIoU, train_loss, valid_acc, valid_MIoU, valid_loss):
    plt.figure()
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.title('loss curves')
    plt.xlabel('epochs')
    plt.ylabel('binary cross entropy loss')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.ylim([0,100])
    plt.plot(train_MIoU, label='train')
    plt.plot(valid_MIoU, label='valid')
    plt.title('MIoU curves')
    plt.xlabel('epochs')
    plt.ylabel('MIoU')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.ylim([0,100])
    plt.plot(train_acc, label='train')
    plt.plot(valid_acc, label='valid')
    plt.title('Acc curves')
    plt.xlabel('epochs')
    plt.ylabel('Mean Pixel Acc')
    plt.legend()
    plt.show()
    
    
    
# plot the images with masks
# img, label, pred are torch tensors
def plot_image_mask(nclasses, img, label, pred=None):
    plt.figure(figsize=(15, 15))
    
    # convert to numpy
    img = img.numpy().transpose(1,2,0)
    label = label.numpy().squeeze()
    if not pred is None:
        pred = pred.numpy().squeeze()
    
    # normalize image to [0, 255]
    min_val, max_val = np.min(img), np.max(img)
    img = (img-min_val) / (max_val-min_val) * 255
    img = img.astype(np.uint8)

    if pred is None:
        # plotting img
        plt.subplot(1, 2, 1)
        plt.title("Input Image")
        plt.imshow(img)
        plt.axis('off')
        
        # plotting label
        plt.subplot(1, 2, 2)
        plt.title("True Mask")
        plt.imshow(label, vmin=0, vmax=nclasses)
        plt.axis('off')
        
    
    else:
        # plotting img
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(img)
        plt.axis('off')
        
        # plotting label
        plt.subplot(1, 3, 2)
        plt.title("True Mask")
        plt.imshow(label, vmin=0, vmax=nclasses)
        plt.axis('off')
        
        # plotting prediction
        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(pred, vmin=0, vmax=nclasses)
        plt.axis('off')
    
    plt.show()
