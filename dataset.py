""" The dataset loader for the building data"""
import os
import sys
import numpy as np 
from torch.utils.data.dataset import Dataset 
import torch
from torchvision import transforms

import cv2


class BuildingDataset(Dataset):
    def __init__(self,dataset="train",augment=False):
        self.dataset = dataset
        self.BASEDIR = os.path.dirname(os.path.abspath(__file__))
        self.DATASETDIR = os.path.join(self.BASEDIR,"dataset",self.dataset)
        self.image_list = (os.listdir(self.DATASETDIR))
        self.input_size = 128
        self.augment = augment
       

    def __len__(self):
        #assert(len(self.image_list)==len(self.label_list))
        
        return len(self.image_list)
    
    def __getitem__(self,index):
        
        image = cv2.imread(os.path.join(self.DATASETDIR,self.image_list[index]))
        image = cv2.resize(image,(self.input_size,self.input_size))
        #image = self.transform(image.astype(np.uint8)).astype(np.float)
        if self.dataset =="train" :
            image = preprocess(image, self.input_size, self.augment)
           
        else:
            image = preprocess(image, self.input_size, False)
            

        if self.dataset !="test" :
            label = self.image_list[index].split("_")[-1].split(".")[0]
            
            label = torch.tensor(int(label)).float()
        else :
            label = torch.tensor([0])
        return image,label

def preprocess(image, input_size, augmentation=True):
    if augmentation:
        crop_transform = transforms.Compose([
            transforms.Resize(input_size // 4 * 5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop(input_size)])
            #,
            #transforms.RandomRotation(10)])
    else:
        crop_transform = transforms.CenterCrop(input_size)

    result = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.45641004,0.43316785 ,0.40853815],[0.24733209 ,0.24242754 ,0.24084231])
    ])(image)
    return result

if __name__ == "__main__":
    
    data = BuildingDataset(dataset="train")
    data[1]