import matplotlib.pyplot as plt



# plot the loss and acc curves
def plot_training_curves(train_acc, train_MIoU, train_loss, valid_acc, valid_MIoU, valid_loss):
    plt.figure()
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.title('loss curves')
    plt.xlabel('epochs')
    plt.ylabel('binary cross entropy loss')
    plt.legend()
    
    plt.figure()
    plt.plot(train_MIoU, label='train')
    plt.plot(valid_MIoU, label='valid')
    plt.title('MIoU curves')
    plt.xlabel('epochs')
    plt.ylabel('MIoU')
    plt.legend()
    
    plt.figure()
    plt.plot(train_acc, label='train')
    plt.plot(valid_acc, label='valid')
    plt.title('Acc curves')
    plt.xlabel('epochs')
    plt.ylabel('Mean Pixel Acc')
    plt.legend()
    
    
    
# plot the images with masks
def plot_image_mask(img, label, nclasses, prediction=None):
    plt.figure(figsize=(15, 15))
    
    if prediction is None:
        # plotting img
        plt.subplot(1, 2, 1)
        plt.title("Input Image")
        plt.imshow(img.numpy().transpose(1,2,0))
        plt.axis('off')
        
        # plotting label
        plt.subplot(1, 2, 2)
        plt.title("True Mask")
        plt.imshow(label.numpy().squeeze(), vmin=0, vmax=nclasses)
        plt.axis('off')
        
    
    else:
        # plotting img
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(img.numpy().transpose(1,2,0))
        plt.axis('off')
        
        # plotting label
        plt.subplot(1, 3, 2)
        plt.title("True Mask")
        plt.imshow(label.numpy().squeeze(), vmin=0, vmax=nclasses)
        plt.axis('off')
        
        # plotting prediction
        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(prediction.numpy().squeeze(), vmin=0, vmax=nclasses)
        plt.axis('off')
    
