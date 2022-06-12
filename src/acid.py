from __future__ import absolute_import, division, print_function

import os
os.environ['KERAS_BACKEND'] = 'tensorflow' 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.disable_eager_execution()
import sys
import random
import math
import re
import time
import numpy as np
import pandas as pd
import cv2
from src.config import Config
import src.utils as utils
import src.model as modellib
from src.model import log
import glob
import PIL.ImageOps   
from PIL import Image 
from astropy.io import fits

sess = tf.Session()
import copy
import collections
from src.utils import rgb_clahe_justl
from planetaryimage import PDS3Image
tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)
Image.MAX_IMAGE_PIXELS = None



#################################################################


#!/usr/bin/env python
"""Input Image Dataset Generator Functions

Functions for generating input and target image datasets from Lunar digital
elevation maps and crater catalogues.
"""


class MainConfig(Config):
    ### Configurations
    
    # Give the configuration a recognizable name
    NAME = "Main"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2#1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 600 // (IMAGES_PER_GPU * GPU_COUNT)

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 70 // (IMAGES_PER_GPU * GPU_COUNT)
    
    LEARNING_RATE = 0.01 
    USE_MINI_MASK = True 
    MAX_GT_INSTANCES = 100




class InferenceConfig(MainConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1 # defines batch size in practice

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 500

    # ADJUST MEAN AS NEEDED
    MEAN_PIXEL = [165.32, 165.32, 165.32]
    
    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.0 #9


def Read_Preprocess_Image(directory, NORMALIZE=1,CONV_GS=1,INVERSE=0, EQUALIZE=0, CLAHE=1, RESIZE=1,LIMITS=[0,-1,0,-1] ):

    if ('.IMG' in directory):
        image_file = PDS3Image.open(directory)
        image_data = image_file.image
        im = image_data
        imagebase =  np.array(im)[LIMITS[0]:LIMITS[1],LIMITS[2]:LIMITS[3]]
    else:
        if (('.FITS' in directory) | ('.FIT' in directory)):
            image_data = fits.getdata(directory, ext=0)
            im = image_data
            imagebase =  np.array(im)[LIMITS[0]:LIMITS[1],LIMITS[2]:LIMITS[3]]
        else:
            im = Image.open(directory)
            imagebase =  np.array(im)[LIMITS[0]:LIMITS[1],LIMITS[2]:LIMITS[3]]

    if (NORMALIZE==1):
        imagebase = imagebase/(np.max(imagebase))
        imagebase = (imagebase*255).astype('uint8')
    if (CONV_GS==1):
        imagebase=np.array(PIL.ImageOps.grayscale(Image.fromarray(imagebase)))
    if (INVERSE==1):
        imagebase= np.array(PIL.ImageOps.invert(Image.fromarray(imagebase)))
    if (EQUALIZE==1):
        imagebase=np.array(PIL.ImageOps.equalize(Image.fromarray(imagebase)))
    if (RESIZE==1):
        imagebase = cv2.resize(np.array(imagebase), (512, 512), interpolation=cv2.INTER_LINEAR) 
    if (CONV_GS==1):
        imagebase = np.stack((imagebase,)*3, axis=-1)
    if (CLAHE==1):
        clahe_l=rgb_clahe_justl(imagebase) # CLahe just L band  ## Improve contrast                                               
        imagebase[...,0]=clahe_l
        imagebase[...,1]=clahe_l
        imagebase[...,2]=clahe_l
    return imagebase



################################################ Beginning of do not change anything here ##############################
# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


################################################ End of do not change anything here #########################################    

class_names = ['BackGround', 'crater']   




def model_inference(image,models_directory,which_models='all'):

    print("""\
--------------------------------------------------        
                                                  
       db         ,ad8888ba,   88  88888888ba,    
      d88b       d8a.    `a8b  88  88      `a8b   
     d8.`8b     d8.            88  88        `8b  
    d8.  `8b    88             88  88         88  
   d8YaaaaY8b   88             88  88         88  
  d8aaaaaaaa8b  Y8,            88  88         8P  
 d8.        `8b  Y8a.    .a8P  88  88      .a8P   
d8.          `8b  `aY8888Ya.   88  88888888Ya.    

--------------------------------------------------                                                         
                    """)




    all_models_array = np.array(sorted(glob.glob(models_directory)))
    if (which_models == 'all'):
        model_path_array = all_models_array
    else:
        model_path_array = all_models_array[which_models]


    Craters_Master_list = []
    for model_path in model_path_array:

        #print ('Loading model from: ' model_path)

        inference_config = InferenceConfig()
        # Recreate the model in inference mode
        model_infer2 = modellib.MaskRCNN(mode="inference", 
                                  config=inference_config,
                                  model_dir=MODEL_DIR)

        ## Load trained weights (fill in path to trained weights here)
        assert model_path != "", "Provide path to trained weights"
        print("Loading weights from ", model_path)
        model_infer2.load_weights(model_path, by_name=True)

        r = model_infer2.detect([image], verbose=2)[0]   # Main inference command

        for roisit in range (0,len(r["rois"])):
            try:
                if (len((np.ravel(r["masks"].T[roisit]))\
                        [(np.ravel(r["masks"].T[roisit])) > 0] ) < 1.*(50120*50120)):#0.5*262144):
                    _, contours, hierarchy = \
                    cv2.findContours(np.transpose(r["masks"])[roisit].copy() ,\
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cnt = contours[0]
                    (__, __), (MA, ma), angle = cv2.fitEllipse(cnt) 
                    if ((ma/(MA+1e-20)) < 9999. ):
                        coords=np.ndarray(shape=(3,1), dtype=float)
                        rois = r["rois"][roisit]

                        coords[2] =  0.5*abs(rois[0]-rois[2])/2. + \
                                     0.5*abs(rois[1]-rois[3])/2.
                        xpixglob =     abs(min(rois[1],rois[3]) + coords[2])
                        ypixglob =     abs(min(rois[0],rois[2]) + coords[2])


                        Craters_Master_list.append\
                        ([float(xpixglob),float(ypixglob),float(coords[2]),\
                            rois[0],rois[1],rois[2],rois[3],r["scores"][roisit],\
                            float(ma/(MA+1e-20)),r["masks"].T[roisit].T, len((np.ravel(r["masks"].T[roisit]))\
                        [(np.ravel(r["masks"].T[roisit])) > 0] )])
            except:
                continue


    return np.array(Craters_Master_list)



def model_inference_grid(directory,models_directory,which_models='all',window_size=512,\
                            NORMALIZE=1,CONV_GS=1,INVERSE=0, EQUALIZE=0, CLAHE=1, RESIZE=1,LIMITS=[0,-1,0,-1] ):



    print("""\
--------------------------------------------------        
                                                  
       db         ,ad8888ba,   88  88888888ba,    
      d88b       d8a.    `a8b  88  88      `a8b   
     d8.`8b     d8.            88  88        `8b  
    d8.  `8b    88             88  88         88  
   d8YaaaaY8b   88             88  88         88  
  d8aaaaaaaa8b  Y8,            88  88         8P  
 d8.        `8b  Y8a.    .a8P  88  88      .a8P   
d8.          `8b  `aY8888Ya.   88  88888888Ya.    

--------------------------------------------------                                                         
                    """)



    if ('.IMG' in directory):
        image_file = PDS3Image.open(directory)
        image_data = image_file.image
        im = image_data
        image_glob =  np.array(im)[LIMITS[0]:LIMITS[1],LIMITS[2]:LIMITS[3]]
    else:
        if (('.FITS' in directory) | ('.FIT' in directory)):
            image_data = fits.getdata(directory, ext=0)
            im = image_data
            image_glob =  np.array(im)[LIMITS[0]:LIMITS[1],LIMITS[2]:LIMITS[3]]
        else:
            im = Image.open(directory)
            image_glob =  np.array(im)[LIMITS[0]:LIMITS[1],LIMITS[2]:LIMITS[3]]


    #print (np.shape(image_glob))

    all_models_array = np.array(sorted(glob.glob(models_directory)))
    if (which_models == 'all'):
        model_path_array = all_models_array
    else:
        model_path_array = all_models_array[which_models]



    Craters_Master_list = []

    M = window_size
    N = window_size
    for model_path in model_path_array:
        for xc in range(0,image_glob.shape[1],M):
            for yc in range(0,image_glob.shape[0],N):  
                try:


                    imagebase = copy.deepcopy(image_glob[yc:yc+window_size,xc:xc+window_size])
                    #print (np.shape(imagebase),np.unique(imagebase))

                    if (NORMALIZE==1):
                        imagebase = imagebase/(np.max(imagebase))
                        imagebase = (imagebase*255).astype('uint8')
                    if (CONV_GS==1):
                        imagebase=np.array(PIL.ImageOps.grayscale(Image.fromarray(imagebase)))
                    if (INVERSE==1):
                        imagebase= np.array(PIL.ImageOps.invert(Image.fromarray(imagebase)))
                    if (EQUALIZE==1):
                        imagebase=np.array(PIL.ImageOps.equalize(Image.fromarray(imagebase)))
                    if (RESIZE==1):
                        imagebase = cv2.resize(np.array(imagebase), (512, 512), interpolation=cv2.INTER_LINEAR) 
                    if (CONV_GS==1):
                        imagebase = np.stack((imagebase,)*3, axis=-1)
                    if (CLAHE==1):
                        clahe_l=rgb_clahe_justl(imagebase) # CLahe just L band  ## Improve contrast                                               
                        imagebase[...,0]=clahe_l
                        imagebase[...,1]=clahe_l
                        imagebase[...,2]=clahe_l

                    image = imagebase
                    #print (np.shape(image),np.unique(image))

                    inference_config = InferenceConfig()
                    # Recreate the model in inference mode
                    model_infer2 = modellib.MaskRCNN(mode="inference", 
                                              config=inference_config,
                                              model_dir=MODEL_DIR)

                    ## Load trained weights (fill in path to trained weights here)
                    assert model_path != "", "Provide path to trained weights"
                    print("Loading weights from ", model_path)
                    model_infer2.load_weights(model_path, by_name=True)

                    r = model_infer2.detect([image], verbose=2)[0]   # Main inference command

                    for roisit in range (0,len(r["rois"])):
                        try:
                            if (len((np.ravel(r["masks"].T[roisit]))\
                                    [(np.ravel(r["masks"].T[roisit])) > 0] ) < 1.*(50120*50120)):#0.5*262144):
                                _, contours, hierarchy = \
                                cv2.findContours(np.transpose(r["masks"])[roisit].copy() ,\
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                                cnt = contours[0]
                                (__, __), (MA, ma), angle = cv2.fitEllipse(cnt) 
                                if ((ma/(MA+1e-20)) < 9999. ):
                                    coords=np.ndarray(shape=(3,1), dtype=float)
                                    rois = r["rois"][roisit]

                                    coords[2] =  (window_size/512)*(0.5*abs(rois[0]-rois[2])/2. + \
                                                 0.5*abs(rois[1]-rois[3])/2.)

                                    xpixglob =  xc + (window_size/512)*abs(min(rois[1],rois[3]) + (0.5*abs(rois[0]-rois[2])/2. + \
                                                 0.5*abs(rois[1]-rois[3])/2.))
                                    ypixglob =  yc + (window_size/512)*abs(min(rois[0],rois[2]) + (0.5*abs(rois[0]-rois[2])/2. + \
                                                 0.5*abs(rois[1]-rois[3])/2.))

                                   # print (np.shape(global_mask),np.shape(r["masks"].T[roisit].T))
                                    global_mask = np.zeros([image_glob.shape[0],image_glob.shape[1]])

                                    resized_masked=np.array(cv2.resize(np.array(r["masks"].T[roisit].T),\
                                        (window_size, window_size), \
                                            interpolation=cv2.INTER_LINEAR))

                                    resized_masked[resized_masked>0] == 255

                                    global_mask[yc:yc+window_size,xc:xc+window_size] =\
                                                    resized_masked


                                    Craters_Master_list.append\
                                    (   [float(xpixglob),float(ypixglob),float(coords[2]),\
                                        yc + (window_size/512)*rois[0], xc + (window_size/512)*rois[1],\
                                        yc + (window_size/512)*rois[2], xc + (window_size/512)*rois[3],\
                                        r["scores"][roisit],float(ma/(MA+1e-20)),\
                                        global_mask, \
                                       # cv2.resize(np.array(r["masks"].T[roisit].T),\
                                       #  (image_glob.shape[1], image_glob.shape[1]), \
                                       #     interpolation=cv2.INTER_LINEAR),\
                                        len((np.ravel(global_mask))\
                                        [(np.ravel(global_mask)) > 0] )])
                        except:
                            raise
                except:
                    raise
    return np.array(Craters_Master_list)





def get_unique_iou(Craters_Master_list,iou_thres=0.5,detection_thres=0.):
    def bb_intersection_over_union(boxA, boxB):
        # determinimume the (x, y)-coordinates of the intersection rectangle
        xA = np.maximum(boxA[0], boxB[0])
        yA = np.maximum(boxA[1], boxB[1])
        xB = np.minimum(boxA[2], boxB[2])
        yB = np.minimum(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / (boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    Craters_Master_list2=pd.DataFrame(copy.deepcopy(Craters_Master_list))
    Craters_Master_list2=Craters_Master_list2[Craters_Master_list2[7] >= detection_thres]
    Craters_Master_list2=Craters_Master_list2.reset_index(drop=True)

    objects_unique = np.ones([11])
    for i in range (0, len(Craters_Master_list2)):
        if len(objects_unique) == 0:
            objects_unique = np.concatenate((objects_unique, \
                                         Craters_Master_list2[i]))
        else :
            Long, Lat, Rad, y1,x1,y2,x2,aa,aa,aa,aa = objects_unique.T
            lo, la, r, Y1,X1,Y2,X2,aa,aa,aa,aa = Craters_Master_list2.T[i]

            iou_index = bb_intersection_over_union( [Y1,X1,Y2,X2],[y1,x1,y2,x2])

            index = (iou_index >= iou_thres) 

            if len(np.where(index == True)[0]) == 0:
                objects_unique = np.vstack((objects_unique, Craters_Master_list2.T[i]))    

    objects_unique=objects_unique[1:,:]

    return np.array(objects_unique)


def get_unique_Longlat(Craters_Master_list,thresh_rad = 1.0,thresh_longlat2 = 1.8,detection_thres=0. ):


    Craters_Master_list2=pd.DataFrame(copy.deepcopy(Craters_Master_list))
    Craters_Master_list2=Craters_Master_list2[Craters_Master_list2[7] >= detection_thres]
    Craters_Master_list2=Craters_Master_list2.reset_index(drop=True)

    objects_unique=np.ones(11)
    for j in range(len(Craters_Master_list2)):
        Long, Lat, Rad,aa,aa,aa,aa,aa,aa,aa,aa = objects_unique.T
        lo, la, r,aa,aa,aa,aa,aa,aa,aa,aa = Craters_Master_list2.T[j]
        la_m = (la + Lat) / 2.
        minr = np.minimum(r, Rad)       # be liberal when filtering dupes

        dL = ((Long - lo)**2 + (Lat - la)**2) / minr**2
        dR = abs(Rad - r) / minr
        index = (dR < thresh_rad) & (dL < thresh_longlat2)

        if len(np.where(index == True)[0]) == 0:
            objects_unique = (np.vstack((objects_unique, Craters_Master_list2.T[j])))
    objects_unique=objects_unique[1:,:]
    return np.array(objects_unique)

