import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pickle
import random
import numpy as np

# system packages
from unet import UNet
from training import model_training, feedforward
from visualization import plot_image_mask



# supports CUDA and MPS
def compute_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


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



# custom dataset class, read images from folder
class SegDataset(Dataset):
    def __init__(self, img_path, class_ind_pair, transform, aug=False):
        self.img_path = img_path
        self.class_ind_pair = class_ind_pair
        self.transform = transform
        self.aug = aug
                
    def __len__(self):
        return len(self.img_path)
    
    
    # convert without normalization and convert to long
    def label_to_tensor(self, img):
        np_img = np.array(img)
        tensor_img = torch.from_numpy(np_img).long()
        return tensor_img
    
    
    def __getitem__(self, idx):
        img_dir = self.img_path[idx]
        
        # reading RGB image and label
        img = Image.open(img_dir).convert('RGB')
        label = Image.open(img_dir.replace('CameraRGB', 'CameraSeg')).split()[0]
        
        # apply transformation
        img = self.transform['img'](img)
        label = self.transform['label'](label)
        
        # convert label to tensor
        label = self.label_to_tensor(label)
        
        # data augmentation (randomly flip p=0.5)
        if self.aug:
            if random.choice([0, 1]) == 1:
                img = torch.flip(img, dims=(-1,))
                label = torch.flip(label, dims=(-1,))
        
        return img, label
    
    

# return the image and label transformation
def get_transform():
    img_transform = transforms.Compose([
        transforms.Resize((432, 576)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    label_transform = transforms.Compose([
        transforms.Resize((432, 576))
    ])
    return {'img':img_transform, 'label':label_transform}



# predict the mask of a sigle image
@torch.no_grad
def inference(model, img):
    model.eval()
    
    device = next(model.parameters()).device
    
    # move data to GPU
    data = img.to(device)
    
    # add the batch dimension
    data = data.unsqueeze(0)
    
    # model inference
    output = model(data).cpu()
    _, pred = torch.max(output.data, 1)
    
    return pred
    


def main():
    # class label to indices mapping
    class_ind_pair = BidirectionalMap()
    classes = [
        'Unlabeled','Building','Fence','Other','Pedestrian', 'Pole', 'Roadline', 
        'Road','Sidewalk', 'Vegetation', 'Car','Wall', 'Traffic sign'
    ]
    nclasses = len(classes)
    for ind, name in enumerate(classes):
        class_ind_pair.add_mapping(ind, name)
    # Save the instance to a pickle file
    with open("class_ind_pair.pkl", "wb") as f:
        pickle.dump(class_ind_pair, f)
    
    # get all image paths
    image_path = []
    dataset = '../Datasets/Semantic Segmentation for Self Driving Cars'
    for root, dirs, files in os.walk(dataset):
        if not 'CameraRGB' in root:
            continue
        for file in files:
            if file.endswith('.png'):
                image_path.append(os.path.join(root, file))
    
    # train-valid-test split on image paths
    random.seed(42)
    random.shuffle(image_path)
    train_path = image_path[:int(0.8*len(image_path))]
    val_path = image_path[int(0.8*len(image_path)):int(0.9*len(image_path))]
    test_path = image_path[int(0.9*len(image_path)):]
    random.seed() # remove seed
    
    # create dataset
    train_dataset = SegDataset(train_path, class_ind_pair, get_transform(), aug=True)
    val_dataset = SegDataset(val_path, class_ind_pair, get_transform())
    test_dataset = SegDataset(test_path, class_ind_pair, get_transform())
    
    # convert to loader object
    train_loader = DataLoader(
        train_dataset, batch_size=4, num_workers=4, pin_memory=True, 
        persistent_workers=True, shuffle=True
    )
    valid_loader = DataLoader(
        val_dataset, batch_size=8, num_workers=4, pin_memory=True, 
        persistent_workers=True, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=8, num_workers=4, pin_memory=True, 
        persistent_workers=True, shuffle=False
    )
    
    # visualizing some examples
    for i in range(0, len(train_dataset), (len(train_dataset)-1)//5):
        x, y = train_dataset[i]
        plot_image_mask(nclasses, x, y)
        
    # define model
    model = UNet(3, nclasses)
    model = model.to(compute_device())
    
    # model training
    model_training(model, train_loader, valid_loader)
    
    # loading the best preforming model
    model.load_state_dict(torch.load(f'{type(model).__name__}.pth'))
    
    # model test preformance
    feedforward(model, test_loader)
    for i in range(0, len(test_dataset), (len(test_dataset)-1)//5):
        x, y = test_dataset[i]
        pred = inference(model, x)
        plot_image_mask(nclasses, x, y, pred)
        
    
    
if __name__ == "__main__":
    main()