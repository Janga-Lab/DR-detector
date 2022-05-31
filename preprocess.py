import os
import PIL
from PIL import Image, ImageFilter
from glob import glob
import numpy as np
import cv2
from scipy.ndimage import zoom
import bm3d

class Preprocess():

    def __init__(self):
        self.gamma = 1.2


    def gammaCorrection(self, img, gamma):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.float64)
        # apply gamma correction using the lookup table
        return cv2.LUT(img, table)

    
    def CLAHE(self, img, clahe):
        return clahe.apply(img)



    def correctBackground(self, img, indices):
        for i in indices:
            img[i[0], i[1]] = 0
        return img

    def imageOriginal(self, image):
        img = cv2.imread(image, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        backgroundMask = np.column_stack(np.where(thresh == 0))

        #gb_img = cv2.GaussianBlur(img, (0, 0), 5)
        #img = cv2.addWeighted(img, 4, gb_img, -4, 128)
        #img = self.channelCLAHE(img)
        green = img[:,:,1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        claheEqualized = self.CLAHE(green, clahe)
        gamma = self.gammaCorrection(claheEqualized, self.gamma)
        output = self.correctBackground(gamma, backgroundMask)
        
        return np.expand_dims(output, axis=-1)


    def channelCLAHE(self, image):
        b,g,r = cv2.split(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        claheBlue = self.CLAHE(b, clahe)
        claheGreen = self.CLAHE(g, clahe)
        claheRed = self.CLAHE(r, clahe)
        claheAllChannels = cv2.merge((claheBlue, claheGreen, claheRed))
        return claheAllChannels

    
    """
    def imageOriginal(self, image):        
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        smooth = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
        g = img[:,:,1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        output = self.CLAHE(g, clahe)
        #output = bm3d.bm3d(output, sigma_psd=0.5, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
        return np.expand_dims(output, axis=-1)
    
    """

