import os
import cv2
import PIL
import pickle
import csv

import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from glob import glob
from configparser import ConfigParser
from loguru import logger

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import KFold

from unet import simple_unet_model
from preprocess import Preprocess
from patches import Patches
from empatches2 import EMPatches


modelMA = "ma_256_ZOOM4.hdf5"
modelEX = "exudate_255_ZOOM_v3.hdf5"
modelHEM = "rh_256_nozoom_gammaV3.hdf5"

DIR = list(sorted(glob("image_directory_here")))


def getCounts(predictedMask):
    gray = cv2.cvtColor(predictedMask, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200)
    contours, hierarchy= cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    numObjs= len(contours)
    return numObjs


def predict(image, model, modelMA, modelEX, modelHEM):
    
    image = Preprocess().imageOriginal(image)

    emp = EMPatches()
    patches, indices = emp.extract_patches(image, patchsize=256, overlap=.2)
    patches = np.array(patches) / 255.

    # microaneurysm counts
    model.load_weights(modelMA)

    pred = model.predict(patches)
    pred = pred > 0.50
    pred = emp.merge_patches(pred, indices)
    pred = pred * 255
    pred = pred.astype(np.uint8)

    countsMA = getCounts(pred)

    
    # exudate counts
    model.load_weights(modelEX)
    
    pred = model.predict(patches)
    pred = pred > 0.50
    pred = emp.merge_patches(pred, indices)
    pred = pred * 255
    pred = pred.astype(np.uint8)
    
    countsEX = getCounts(pred)

    model.load_weights(modelHEM)

    pred = model.predict(patches)
    pred = pred > 0.50
    pred = emp.merge_patches(pred, indices)
    pred = pred * 255
    pred = pred.astype(np.uint8)

    countsHEM = getCounts(pred)

    
    

    return countsMA, countsEX, countsHEM
    



def getDRStatus(sample, status):
    try:
        status = [x[1] for x in status if x[0] == sample][0]
    except IndexError:
        status = "NA"
    return status
    
    
if __name__ == "__main__":

    status = []
    with open("drStatus.txt", "r") as f:
        r = csv.reader(f, delimiter="\t")
        for s in r:
            status.append(s)
            
    
    model = simple_unet_model(256, 256, 1)

    with open("lesions.csv", "w+") as handle:
        handle.write("sample\tmicroaneurysm\texudate\themorrhage\tstatus\n")

        for i in DIR:
            sample = i.split(".")[0]
            drStatus = getDRStatus(sample, status)
            countsMA, countsEX, countsHEM = predict(i, model, modelMA, modelEX, modelHEM)
            handle.write(f"{sample}\t{countsMA}\t{countsEX}\t{countsHEM}\t{status}\n")
    
    

