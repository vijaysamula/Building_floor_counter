"""
Helps in saving the image after prediction from the model.
This file is use in eval.py to visualize the model.
"""
import os 
import sys 
import numpy as np 
#sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
import torch
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import PIL
BASEDIR = os.path.dirname(os.path.abspath(__file__))
DATASETDIR = os.path.join(BASEDIR,"dataset","ground_truth_2011")

def change_file_name():
    """
    The file is intended to change the file names in the dataset folder 
    """
    images_list = os.listdir(DATASETDIR)
    for idx,i in enumerate(images_list):
        image = cv2.imread(os.path.join(DATASETDIR,i))
        print(i)
        dst = "ground2011_"+str(idx)+"_6"+".png"
        #print(dst)
        cv2.imwrite(os.path.join(DATASETDIR,dst),image)


def image_write(img,text,out_file):
    """
    The file is intended to save the image after prediction with text 
    it takes in image , text to write on the image , output_filename to save the image 
    """
    img = img.transpose(2,1,0).transpose(1,0,2)
    
    npimg = (img*np.array([0.24733209 ,0.24242754 ,0.24084231])) + np.array([0.45641004,0.43316785 ,0.40853815])
    
    npimg = npimg*255
    outimg = Image.fromarray(npimg.astype(np.uint8))
    draw = ImageDraw.Draw(outimg)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 10)
    # draw.text((x, y),"Sample Text",(r,g,b))
    draw.text((0, 0),text,(255,0,0),font=font)
    outimg.save(out_file)

    #cv2.waitKey(0)

if __name__=="main":
    change_file_name()