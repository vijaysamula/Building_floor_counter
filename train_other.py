"""
other files to just check different models
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
from loss import RMSELoss
import torch.nn as nn 
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

PATH = './building_net_icons_finetuning2.pth'
writer = SummaryWriter('log/floor_counter_icons')
dataloaders = { x : torch.utils.data.DataLoader(BuildingDataset(dataset=x,augment=True  ), batch_size=6,
                                          shuffle=True)
                for x in ["train","val"]
}
dataloaders["val"] = torch.utils.data.DataLoader(BuildingDataset(dataset="val",augment=False  ), batch_size=6,
                                          shuffle=True)
dataset_sizes = {x: dataloaders[x].__len__() for x in ['train', 'val']}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device using : ",device)
def imshow(img,title=None):
    npimg = img.numpy().transpose(2,1,0)
    print(npimg.shape)
    #npimg =npimg*255
    npimg = np.clip(npimg, 0, 1)

    plt.imshow(npimg)
    
    if title is not None:
        plt.title(title)
    plt.pause(0.001) 
    plt.show()

#images, labels = next(iter(dataloaders['train']))
#out = torchvision.utils.make_grid(images)


def train(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_mse = 0

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
                    loss = torch.sum((outputs.squeeze(1)-labels)**2/weights[torch.clamp(outputs.squeeze(1).long()-15,0,57)])/len(outputs)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                #print(preds,labels)
                #print(preds.shape,labels.shape)
                running_loss += loss.item() * inputs.size(0)
                running_mse  += torch.abs(preds-labels).sum().item()
            if phase == 'train':
                scheduler.step()
            #print("dataset_size",labels,preds)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_mse =  running_mse /  dataset_sizes[phase]
            if phase == "train":    
                writer.add_scalar("trainiing_loss",epoch_loss,epoch*dataset_sizes[phase])
                writer.add_scalar("mse_error_train",epoch_mse,epoch*dataset_sizes[phase])
            else :
                 writer.add_scalar("validation_loss",epoch_loss,epoch*dataset_sizes[phase])
                 writer.add_scalar("mse_error_val",epoch_mse,epoch*dataset_sizes[phase])
            #epoch_error = running_mse.do / dataset_sizes[phase]

            print('{} Loss: {:.4f} mse_error : {:.4f}'.format(
                phase, epoch_loss , epoch_mse))

            # deep copy the model
            if phase == 'val' :
                
                best_model_wts = copy.deepcopy(model.state_dict())
                model.load_state_dict(best_model_wts)
                torch.save(model.state_dict(), PATH)

    

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    

  
    
    # load best model weights
    
    return model

def visualize_model( type_train="model_ft",num_images=6):
    if type_train=="model_conv":
        # model_conv = torchvision.models.resnet18(pretrained=True)
        # for param in model_conv.parameters():
        #     param.requires_grad = False
        
        # # Parameters of newly constructed modules have requires_grad=True by default
        # num_ftrs = model_conv.fc.in_features
        # model_conv.fc = nn.Linear(num_ftrs, 1)
        model_conv = GenderAge(1,[2,2,2,2])
        net = model_conv.to(device)
    elif type_train=="model_ft":
        model_ft = torchvision.models.resnet18(pretrained=True)
        
        
        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 1)
        net = model_ft.to(device)
        # model_ft = FloorModel()
        # # num_ftrs = model_ft.fc.in_features
        # # model_ft.fc = nn.Linear(num_ftrs, 1)

        # net = model_ft.to(device)
    else:
        return

    
    was_training = net.training
    
    net.load_state_dict(torch.load(PATH))
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            preds = torch.clamp(outputs.squeeze(1),0,50)   # assuming 50 floors max

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                mse = torch.abs(preds[j]-labels[j]).sum().item()
                print("mean square error :",mse)
                ax.set_title('predicted: {} Floors , mse : {}'.format(torch.round(preds[j]),mse))
                
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    net.train(mode=was_training)
                    return
        net.train(mode=was_training)

def fine_tuning():

    model_ft = FloorModel()
    # # num_ftrs = model_ft.fc.in_features
    # # model_ft.fc = nn.Linear(num_ftrs, 1)

    model_ft = model_ft.to(device)
    # model_ft = torchvision.models.resnet18(pretrained=True)  
    # # Parameters of newly constructed modules have requires_grad=True by default
    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, 1)
    # model_ft = model_ft.to(device)

    optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=0.002,weight_decay=5e-5)
    criterion = RMSELoss()
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model_ft.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model_ft.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.7)

    model_ft = train(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=180)
    return model_ft

def feature_extractor():
    # model_conv = torchvision.models.resnet18(pretrained=True)
    # for param in model_conv.parameters():
    #     param.requires_grad = False
    
    # # Parameters of newly constructed modules have requires_grad=True by default
    # num_ftrs = model_conv.fc.in_features
    # model_conv.fc= nn.Linear(num_ftrs, 1)

    model_conv = GenderAge(1,[2,2,2,2])
    
    

    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model_conv.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model_conv.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')



    model_conv = model_conv.to(device)
    optimizer_conv = torch.optim.Adam(model_conv.parameters(), lr=0.0001,weight_decay=5e-5)
    criterion = torch.nn.MSELoss() 
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=20, gamma=0.7)
    model_conv = train(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=180)

    return model_conv
def rmse_metric(pred,label):
    return torch.sqrt(torch.mean(pred,label))
    
if __name__ == "__main__":
    model_conv= feature_extractor()
    #model_ft = fine_tuning()
    visualize_model(type_train="model_conv")
