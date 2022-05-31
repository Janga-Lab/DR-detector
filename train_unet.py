\
"""
trainBinaryUnet.py


"""
import os
import cv2
import PIL
import pickle

import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from glob import glob
from PIL import Image
from configparser import ConfigParser
from loguru import logger

from tensorflow.keras.utils import normalize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import KFold

from unet import simple_unet_model
from preprocess import Preprocess
from patches import Patches
from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_edt
import random
import imgaug as ia
import imgaug.augmenters as iaa

class training():

    def __init__(self):
        pass


    def getDataset(self, dir1, dir2):
        d = []
        for i in os.listdir(dir1):
            n = i.split("_")[0]
            #print(i)
            for j in os.listdir(dir2):
                if j.split("_")[0] == n:
                    d.append([dir1+i, dir2+j])
        return d

    
    
    def getModel(self):
        return simple_unet_model(256, 256, 1)


    def unet_weight_map(self, y, wc=None, w0 = 10, sigma = 5):
        y = np.squeeze(y, axis=-1)
        labels = label(y)
        no_labels = labels == 0
        label_ids = sorted(np.unique(labels))[1:]

        if len(label_ids) > 1:
            distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)))

            for i, label_id in enumerate(label_ids):
                distances[:,:,i] = distance_transform_edt(labels != label_id)

            distances = np.sort(distances, axis=2)
            d1 = distances[:,:,0]
            d2 = distances[:,:,1]
            w = w0 * np.exp(-1/2*((d1 + d2) / sigma)**2) * no_labels

            if wc:
                class_weights = np.zeros_like(y)
                for k, v in wc.items():
                    class_weights[y == k] = v
                w = w + class_weights
        else:
            w = np.zeros_like(y)

        return w
    


    def paddedZoom(self, img, zoomfactor=2.0):
        
        '''
        Zoom in/out an image while keeping the input image shape.
        i.e., zero pad when factor<1, clip out when factor>1.
        there is another version below (paddedzoom2)
        '''

        out  = np.zeros_like(img)
        zoomed = np.expand_dims(cv2.resize(img, None, fx=zoomfactor, fy=zoomfactor), axis=-1)

        h, w, c = img.shape
        zh, zw, c = zoomed.shape

        if zoomfactor<1:    # zero padded
            out[round((h-zh)/2):-round((h-zh)/2), round((w-zw)/2):-round((w-zw)/2)] = zoomed
        else:               # clip out
            out = zoomed[round((zh-h)/2):-round((zh-h)/2), round((zw-w)/2):-round((zw-w)/2)]
        
        return out



    
    def _exec(self):
        #data = list(zip(sorted(glob("/N/project/Mammalian_Genomics/dr/glaucoma/Drishti-GS1_files/data/Images/*.png")), sorted(glob("/N/project/Mammalian_Genomics/dr/glaucoma/Drishti-GS1_files/data/GT/cup/*.png"))))[:25]
        #data =list(zip(sorted(glob("/N/slate/hungill/nonvessel/idridSeg/images/img/*.jpg")), sorted(glob("/N/slate/hungill/nonvessel/idridSeg/GT/train/hardExudateBW/*.tif"))))[:1]
        #data = list(zip(sorted(glob('/N/slate/hungill/nonvessel/lesion/preretinalHemorrhageIMG/*.jpg')),sorted(glob('/N/slate/hungill/nonvessel/lesion/preretinalMyMasks/*.png')))) 
        #data = list(zip(sorted(glob('/N/slate/hungill/nonvessel/lesion/retinalHemorrhageIMG/*.jpg')),sorted(glob('/N/slate/hungill/nonvessel/lesion/retinalHemorrhageGT/*.png'))))
        data = self.getDataset('/N/slate/hungill/nonvessel/lesion/retinalHemorrhageIMG/','/N/slate/hungill/nonvessel/lesion/retinalMyMasks/')# + self.getDataset('/N/slate/hungill/nonvessel/lesion/preretinalHemorrhageIMG/','/N/slate/hungill/nonvessel/lesion/preretinalMyMasks')
        #data = list(zip(sorted(glob('/N/slate/hungill/nonvessel/lesion/retinalHemorrhageIMG/*.jpg')),sorted(glob('/N/slate/hungill/nonvessel/lesion/retinalMyMasks/*.png')))) 
        #data = list(zip(sorted(glob('/N/project/Mammalian_Genomics/dr/e_optha_MA/microaneurysmIMG/*.jpg')),sorted(glob('/N/project/Mammalian_Genomics/dr/e_optha_MA/microaneurysmGT/*.png'))))[:300]
        #data =list(zip(sorted(glob('/N/slate/hungill/nonvessel/idridSeg/images/img/*.jpg')),sorted(glob('/N/slate/hungill/nonvessel/idridSeg/GT/train/microaneurysm/*.tif'))))[:25]
        #data = list(zip(sorted(glob('/N/slate/hungill/nonvessel/lesion/retinalHemorrhageIMG/*.jpg')),sorted(glob('/N/slate/hungill/nonvessel/lesion/retinalMyMasks/*.png'))))
        #data = self.getDataset('/N/slate/hungill/nonvessel/lesion/retinalHemorrhageIMG/','/N/slate/hungill/nonvessel/lesion/retinalMyMasks/')
        #data = list(zip(sorted(glob('/N/slate/hungill/nonvessel/lesion/fibrousProliferationIMG/*.jpg')),sorted(glob('/N/slate/hungill/nonvessel/lesion/fibrousProliferationGT/*.png'))))
        #data = list(zip(sorted(glob('/N/project/Mammalian_Genomics/dr/e_optha_EX/exudateIMG/*.jpg')), sorted(glob('/N/project/Mammalian_Genomics/dr/e_optha_EX/exudateGT/*.png'))))
        #data = list(zip(sorted(glob('/N/project/Mammalian_Genomics/dr/lesions/hardExudateIMG/*.jpg')),sorted(glob('/N/project/Mammalian_Genomics/dr/lesions/hardExudateGT/*.png'))))
        #data = list(zip(sorted(glob("/N/slate/hungill/nonvessel/idridSeg/images/img/*.jpg")), sorted(glob("/N/slate/hungill/nonvessel/idridSeg/GT/train/hemorrhage/*.tif"))))[0:25]

        """
        preprocess retina fondus images
        divide retina fondus images and ground truth segmentation masks into patches
        """
        
        allImagePatches = []
        allGroundTruthPatches = []

        for i in data:
            print(i)
            image = Preprocess().imageOriginal(i[0])
            ground_truth = np.expand_dims(cv2.imread(i[1], 0), axis=-1)
            imagePatches, groundtruthPatches = Patches()._positivePatches(image, ground_truth, 256)
            for j in imagePatches:
                #allImagePatches.append(self.paddedZoom(j))
                allImagePatches.append(j)
                
            for j in groundtruthPatches:
                #allGroundTruthPatches.append(self.paddedZoom(j))
                allGroundTruthPatches.append(j)
        
        allImagePatches = np.array(allImagePatches) #/ 255.
        allGroundTruthPatches = np.array(allGroundTruthPatches) #/ 255.

        
        
        from sklearn.model_selection import train_test_split
        x_train, x_val, y_train, y_val = train_test_split(allImagePatches, allGroundTruthPatches, test_size=0.2, random_state=0)

        print(x_train.shape)
        print(y_train.shape)
        
        seed = 1

        data_gen_args = dict(rescale=1./255)
        
        trainImgGen = ImageDataGenerator(**data_gen_args)
        trainGtGen = ImageDataGenerator(**data_gen_args)
        valImgGen = ImageDataGenerator(**data_gen_args)
        valGtGen = ImageDataGenerator(**data_gen_args)
        
        trainImgGen.fit(x_train, augment=True, seed=seed)
        trainGtGen.fit(y_train, augment=True, seed=seed)
        valImgGen.fit(x_val, augment=True, seed=seed)
        valGtGen.fit(y_val, augment=True, seed=seed)
        
        
        trainImgGenerator = trainImgGen.flow(x_train, seed=seed)
        trainGtGenerator = trainGtGen.flow(y_train, seed=seed)
        valImgGenerator = valImgGen.flow(x_val, seed=seed)
        valGtGenerator = valGtGen.flow(y_val, seed=seed)

    
        trainGen = zip(trainImgGenerator, trainGtGenerator)
        valGen = zip(valImgGenerator, valGtGenerator)
        
        
        """
        next(trainGen)
        #next(trainGen)
        a, b = next(trainGen)

        idx = 0
        for i, j in zip(a, b):
            idx += 1
            margin=50 # pixels
            dpi = float(plt.rcParams['figure.dpi'])

            f, (p1, p2) = plt.subplots(1, 2, figsize=(5000/dpi,5000/dpi))

            p1.set_title("Image", fontsize=45, fontweight="medium")
            p1.imshow(np.squeeze(i, axis=-1), cmap='gray')
            p1.axis('off')

            p2.set_title("GT", fontsize=45, fontweight="medium")
            p2.imshow(np.squeeze(j, axis=-1), cmap='gray')
            p2.axis('off')


            f.subplots_adjust(wspace=5/dpi, hspace=0)
            f.savefig(f'{idx}_idrid.png')
        
        """
        
        model = self.getModel()

        checkpoint = ModelCheckpoint("rh_256_nozoom_gammaV3.hdf5", monitor='val_iou_m', verbose=1, save_best_only=True, mode='max')
        earlyStop = EarlyStopping(monitor ="val_iou_m", mode ="max", patience = 25, restore_best_weights = True)
        modelCallbacks = [checkpoint, earlyStop]

        batchSize = 32
        steps = len(x_train) // batchSize
        valSteps = len(x_val) // batchSize

        history = model.fit(trainGen, validation_data=valGen, validation_steps=valSteps, epochs=200, steps_per_epoch=steps, callbacks=modelCallbacks)
        tf.keras.backend.clear_session()
        
        
training()._exec()
