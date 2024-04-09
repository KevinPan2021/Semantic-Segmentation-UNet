import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torchvision import transforms
import pickle
import time
import threading
import random

# system packages
from model import UNet
from training import model_training, feedforward
from visualization import plot_image_mask



# supports MacOS mps and CUDA
def GPU_Device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'


# bidirectional dictionary
class BidirectionalMap:
    def __init__(self):
        self.key_to_value = {}
        self.value_to_key = {}
    
    def __len__(self):
        return len(self.key_to_value)
    
    def add_mapping(self, key, value):
        self.key_to_value[key] = value
        self.value_to_key[value] = key

    def get_value(self, key):
        return self.key_to_value.get(key)

    def get_key(self, value):
        return self.value_to_key.get(value)



# predict the mask of a sigle image
def inference(model, device, img):
    # move data to GPU
    data = img.to(device)
    
    # add the batch dimension
    data = data.unsqueeze(0)
    
    # model inference
    with torch.no_grad():  # Disable gradient calculation
        output = model(data).cpu()
    _, predicted = torch.max(output.data, 1)
    
    return predicted
    
    
# reading image using parallel processing
def parallel_read_images(path, X, transform, num_threads=8):
    
    def read_image(path, data, transform, start_idx, end_idx):
        for img_file in os.listdir(path)[start_idx:end_idx]:
            if img_file.endswith('DS_Store'):
                continue
            img = Image.open(os.path.join(path, img_file)).convert('RGB')
            data[f"{path.split('/')[-2]}_{img_file}"] = transform(img)

    # Split the data indices into equal parts for each thread
    total_size = len(os.listdir(path))
    chunk_size = total_size // num_threads
    chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_threads)]
    chunks[-1] = (chunks[-1][0], total_size)  # Adjust the last chunk to include remaining indices

    threads = []
    for start_idx, end_idx in chunks:
        thread = threading.Thread(target=read_image, args=(path, X, transform, start_idx, end_idx))
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    return X
        
        

# reading image using parallel processing
def parallel_read_masks(path, Y, transform, num_threads=8):
    
    def read_mask(path, data, transform, start_idx, end_idx):
        for label_file in os.listdir(path)[start_idx:end_idx]:
            if label_file.endswith('DS_Store'):
                continue
            label = Image.open(os.path.join(path, label_file)).split()[0]
            #data.append(transform(label))
            data[f"{path.split('/')[-2]}_{label_file}"] = transform(label)

    # Split the data indices into equal parts for each thread
    total_size = len(os.listdir(path))
    chunk_size = total_size // num_threads
    chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_threads)]
    chunks[-1] = (chunks[-1][0], total_size)  # Adjust the last chunk to include remaining indices

    threads = []
    for start_idx, end_idx in chunks:
        thread = threading.Thread(target=read_mask, args=(path, Y, transform, start_idx, end_idx))
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    return Y


    
# convert from numpy array to pytorch tensor object
def convert_to_tensor(X, Y):
    X = torch.stack(X)
    Y = torch.stack(Y)
    return X, Y



def train_test_split(X, Y, thres):
    thres = sorted(thres) # threshold must be in ascending order
    # preform train-valid-test split
    if len(thres) == 2:
        trainX = X[:int(thres[0] * len(X))]
        trainY = Y[:int(thres[0] * len(Y))]
        
        validX = X[int(thres[0] * len(X)):int(thres[1] * len(X))]
        validY = Y[int(thres[0] * len(X)):int(thres[1] * len(Y))]
        
        testX = X[int(thres[1] * len(X)):]
        testY = Y[int(thres[1] * len(Y)):]
    
        del X, Y
        return trainX, trainY, validX, validY, testX, testY
    
    # preform train-test split
    elif len(thres) == 1:
        trainX = X[:int(thres[0] * len(X))]
        trainY = Y[:int(thres[0] * len(Y))]
        
        testX = X[int(thres[0] * len(X)):]
        testY = Y[int(thres[0] * len(Y)):]
    
        del X, Y
        return trainX, trainY, testX, testY
    
    
    
# Concatenate original and flipped data
def data_augmentation(X, Y):
    X = torch.cat([X, torch.flip(X, dims=(3,))])
    Y = torch.cat([Y, torch.flip(Y, dims=(3,))])
    return X, Y
    

def main():
    start_time = time.time()
    
    # dataset directory
    dataset = '../Datasets/Semantic Segmentation for Self Driving Cars'
    
    # class label to indices mapping
    class_ind_pair = BidirectionalMap()
    classes = ['Unlabeled','Building','Fence','Other','Pedestrian', 'Pole', 'Roadline', 
               'Road','Sidewalk', 'Vegetation', 'Car','Wall','Traffic sign']
    nclasses = len(classes)
    for ind, name in enumerate(classes):
        class_ind_pair.add_mapping(ind, name)
    # Save the instance to a pickle file
    with open("class_ind_pair.pkl", "wb") as f:
        pickle.dump(class_ind_pair, f)
    
    # image transform
    img_transform = transforms.Compose([
        transforms.Resize((288, 384)), # original shape 600*800
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((288, 384)),# original shape 600*800
        transforms.Lambda(lambda img: torch.from_numpy(np.array(img)).long())
    ])
    
    # reading data and masks
    X_map, Y_map = dict(), dict() # use dictionary to ignore threading breaking order
    for folder in tqdm(os.listdir(dataset)):
        if folder.endswith('DS_Store') or folder.endswith('.md'):
            continue
        parallel_read_images(f'{dataset}/{folder}/CameraRGB', X_map, img_transform)
        parallel_read_masks(f'{dataset}/{folder}/CameraSeg', Y_map, mask_transform)
    
    # shuffle the dataset and pair X and Y
    X, Y = [], []
    shuffled_keys = list(X_map.keys())
    random.seed(42)
    random.shuffle(shuffled_keys)
    for key in shuffled_keys:
        X.append(X_map[key])
        Y.append(Y_map[key])
    del X_map, Y_map
    
    # visualizing data and mask
    for i in range(0, len(X), 1000):
        plot_image_mask(X[i], Y[i], nclasses)
    

    # convert to tensor
    X, Y = convert_to_tensor(X, Y)
    print('done converting to tensor')
    
    # train-test split
    trainX, trainY, validX, validY, testX, testY = train_test_split(X, Y, [0.8, 0.9])
    print('done train/test split')
    
    # data augmentation on train data
    #trainX, trainY = data_augmentation(trainX, trainY)
    print('done data augmentation')
    
    # convert to loader object
    train_loader = DataLoader(TensorDataset(trainX, trainY), batch_size=16, shuffle=True)
    valid_loader = DataLoader(TensorDataset(validX, validY), batch_size=16, shuffle=True)
    del trainX, trainY, validX, validY
    print('done convert to data loader')
    
    # define model
    model = UNet(3, nclasses)
    model = model.to(GPU_Device())
    
    # model training
    model_training(model, nclasses, train_loader, valid_loader, GPU_Device())
    print('done training')
    
    # loading the best preforming model
    model.load_state_dict(torch.load(f'{type(model).__name__}.pth'))
    print('done loading best model')
    
    # model test preformance
    test_loader = DataLoader(TensorDataset(testX, testY), batch_size=16, shuffle=True)
    test_acc, test_MIoU, test_loss = feedforward(model, nclasses, test_loader, GPU_Device())
    print(f'Test Loss: {test_loss:.3f} | MIoU: {test_MIoU:.2f}%| Acc: {test_acc:.2f}%')
    
    # visualizing the test results
    for i in range(0, len(testX), 100):
        predicted = inference(model, GPU_Device(), testX[i])
        plot_image_mask(testX[i], testY[i], nclasses, predicted)
    del testX, testY
    
    print('took', time.time() - start_time)
    
    
if __name__ == "__main__":
    main()