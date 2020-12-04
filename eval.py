""" 
The eval file to evaluate the validation or test dataset.
Visualization function is also provided here
-------------------------------------------------------------------------------------------------------------------------------
python eval.py --checkpoint_path path/to/checkpoint/folder --dataset path/to/valid/or/test --save_img path/to/prediction/folder
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
from auxilary import image_write

import time
import copy
import torch.nn as nn 
import argparse



parser = argparse.ArgumentParser()


parser.add_argument('--checkpoint_path', default='checkpoints/floor_detection_3_quantile.pth', help='Model checkpoint path')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--dataset',  default="test", help='dataset folder to visulaize image')
parser.add_argument('--save_img',  default="predictions/", help='sto save the predicted images in the specific folder')



FLAGS = parser.parse_args()
testloader = torch.utils.data.DataLoader(BuildingDataset(dataset=FLAGS.dataset), batch_size=FLAGS.batch_size,
                                          shuffle=True)


PATH = FLAGS.checkpoint_path
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
def imshow(img,title=None):
    #print(img.shape)
    npimg = img.numpy().transpose(2,1,0).transpose(1,0,2)
    #print(npimg.shape)
    npimg = (npimg*np.array([0.24733209 ,0.24242754 ,0.24084231])) + np.array([0.45641004,0.43316785 ,0.40853815])
    npimg = np.clip(npimg, 0, 1)
    plt.imshow(npimg)
    if title is not None:
        plt.title(title)
        
    plt.pause(0.001) 
    plt.show()


def visualize_model( num_images=6):
    
    model = FloorModel(1,[2,2,2,2])
    net = model.to(device)
   
    was_training = net.training
    
    net.load_state_dict(torch.load(PATH))
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            preds = torch.clamp(outputs.squeeze(1),0,57)   # assuming 50 floors max

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                
                ax.axis('off')
                
                if FLAGS.dataset != "test":
                    mae = torch.abs(preds[j]-labels[j]).sum().item()
                    print("mean absolute error :",mae)
                    ax.set_title('predicted: {} Floors , mae : {}'.format(torch.round(preds[j]),mae))
                    text = '{} Floors,mae : {}'.format(torch.round(preds[j]),mae)
                    
                else :
                    ax.set_title('predicted: {} Floors '.format(torch.round(preds[j])))
                    text = 'predicted: {} Floors '.format(torch.round(preds[j]))
                img = inputs.cpu().data[j]
                #print(img)
                #img = img.transpose(2,1,0).transpose(1,0,2)
                imshow(img)
                out_file = os.path.join(FLAGS.save_img+"predicted_img_{}_{}".format(j,torch.round(preds[j]))+".png")
                print(out_file)
                image_write(img.numpy(),text,out_file)
                if images_so_far == num_images:
                    net.train(mode=was_training)
                    return
        net.train(mode=was_training)
    
if __name__ == "__main__":
    
    visualize_model()
   
    