"""
the main function to train the model 
to run
python train.py --batch_size 3
"""

import numpy as np
import torch 
import sys
sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
import os
from dataset import BuildingDataset
import matplotlib.pyplot as plt 
import torchvision
from model import FloorModel
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import copy
from loss import criterion
import torch.nn as nn 
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--summary_path', default='log/floor_counter_real_data_quantile_3', help='saummary to visualize in tensor board')
parser.add_argument('--checkpoint_path', default='checkpoints/floor_detection_3_quantile.pth', help='Model checkpoint path')
parser.add_argument('--max_epoch', type=int, default=180, help='Epoch to run [default: 180]')
parser.add_argument('--batch_size', type=int, default=3, help='Batch Size during training [default: 3]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='Optimization L2 weight decay [default: 0]')


FLAGS = parser.parse_args()


PATH = FLAGS.checkpoint_path
writer = SummaryWriter(FLAGS.summary_path)
dataloaders = { x : torch.utils.data.DataLoader(BuildingDataset(dataset=x,augment=True  ), batch_size=FLAGS.batch_size,
                                          shuffle=True,drop_last=True)
                for x in ["train","val"]
}

dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'val']}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The model is training on : ",device)

print("The Training data : ",dataset_sizes["train"])
print("The Validation data : ",dataset_sizes["val"])


def train(model, criterion, optimizer, scheduler, num_epochs=25):
    """
    trains the model and evaluates the validation data
    takes in model, loss,criterion, optimizer, scheduler , num_epochs

    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 100)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_mae = 0.0
            running_mape = 0.0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                weights = torch.zeros(58).cuda()
                for i in range(15, 73):
                    weights[i - 15] = torch.sum((torch.ones_like(labels) * i - labels) ** 2) / len(labels)
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.clamp(outputs.squeeze(1),0,50)
                    #loss = criterion(torch.clamp(outputs.squeeze(1),0,50), labels)
                    #loss = torch.sum((outputs.squeeze(1)-labels)**2/weights[torch.clamp(outputs.squeeze(1).long(),0,57)])/len(outputs)
                    #loss = torch.sum((outputs.squeeze(1)-labels)**2/weights[torch.clamp(outputs.squeeze(1).long()-15,0,57)])/len(outputs) # mse loss
                    loss =  criterion(outputs.squeeze(1),labels)     # Quantile loss
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                
                running_loss += loss.item() * inputs.size(0)
                running_mae  += torch.abs(preds-labels).sum().item()
                running_mape  += torch.abs(((preds-labels)/labels)*100).sum().item()
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss /  dataset_sizes[phase]
            epoch_mae =  running_mae /   dataset_sizes[phase]
            epoch_mape =  running_mape /  dataset_sizes[phase]
            epoch_mae /= FLAGS.batch_size
            epoch_mape /= FLAGS.batch_size
            if phase == "train":    
                writer.add_scalar("trainiing_loss",epoch_loss,epoch*dataset_sizes[phase])
                writer.add_scalar("mae_error_train",epoch_mae,epoch*dataset_sizes[phase])
                writer.add_scalar("mape_error_train",epoch_mape,epoch*dataset_sizes[phase])   # summary writer 
            else :
                writer.add_scalar("validation_loss",epoch_loss,epoch*dataset_sizes[phase])
                writer.add_scalar("mae_error_val",epoch_mae,epoch*dataset_sizes[phase])
                writer.add_scalar("mape_error_val",epoch_mape,epoch*dataset_sizes[phase])

            print('{} Loss: {:.4f} mae_error : {:.4f} mape_error : {:.4f} %'.format(
                phase, epoch_loss , epoch_mae,epoch_mape))

            # deep copy the model
            if phase == 'val' :
                
                best_model_wts = copy.deepcopy(model.state_dict())
                model.load_state_dict(best_model_wts)               # model is saved at every epoch
                torch.save(model.state_dict(), PATH)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    
    return model





def feature_extractor():
    """
        the model function 
    """
    
    model_conv = FloorModel(1,[2,2,2,2])
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model_conv.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model_conv.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    model_conv = model_conv.to(device)
    optimizer_conv = torch.optim.Adam(model_conv.parameters(), lr=FLAGS.learning_rate,weight_decay=FLAGS.weight_decay)
    criterion = torch.nn.MSELoss() 
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=15, gamma=0.7)
    model_conv = train(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=FLAGS.max_epoch)

    return model_conv

    
if __name__ == "__main__":
    

    model_conv= feature_extractor()
    
    
