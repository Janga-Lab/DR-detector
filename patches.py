import numpy as np
from cv2 import imread, flip, imwrite
from patchify import patchify
from random import sample
from PIL import Image
from empatches import EMPatches


class Patches():

    def __init__(self):
        pass


    def readIMG(self, img):
        return imread(img)
        #x = np.array(imread(img))/255.
        #return x


    def _predictPatches(self, img, patchSize, stepSize, channels):
        imgPatchList = []
        imgPatches = patchify(img, (patchSize, patchSize, channels), step=stepSize)
        for i in range(imgPatches.shape[0]):
            for j in range(imgPatches.shape[1]):
                patch = imgPatches[i, j, :, :]
                imgPatchList.append(patch)
        imgPatchList = [np.squeeze(x, axis=0) for x in imgPatchList]
        return np.array(imgPatchList), imgPatches.shape

    def _allPatches2(self, img, gt, patch_size):
        emp = EMPatches()
        imgPatches, imgIndices = emp.extract_patches(img, patchsize=int(patch_size), overlap=0.7)
        gtPatches, gtIndices = emp.extract_patches(gt, patchsize=int(patch_size), overlap=0.7)
        return np.array(imgPatches), np.array(gtPatches)




    def _allPatches(self, img, gt, patchSize):
        #step = 1
        step = round(int(patchSize) * 0.15)
        imgPatchList = []
        gtPatchList = []
        imgPatches = patchify(img, (patchSize, patchSize, 1), step=step)
        gtPatches = patchify(gt, (patchSize, patchSize, 1), step=step)

        for i in range(imgPatches.shape[0]):
            for j in range(imgPatches.shape[1]):
                imgPatchList.append(np.squeeze(imgPatches[i, j, :, :], axis=0))


        for i in range(gtPatches.shape[0]):
            for j in range(gtPatches.shape[1]):
                gtPatchList.append(np.squeeze(gtPatches[i, j, :, :], axis=0))


                
        return imgPatchList, gtPatchList

        
    
    def readGT(self, gt):
        x = np.array(imread(gt))
        return x[..., np.newaxis]/np.max(x)

    
    def _balancedPatches(self, img, gt, patchSize):
        step = round(int(patchSize) * 0.1)
        imgPatchList = []
        gtPatchList = []
        imgPatches = patchify(img, (patchSize, patchSize, 1), step=step)
        gtPatches = patchify(gt, (patchSize, patchSize, 1), step=step)


        for i in range(imgPatches.shape[0]):
            for j in range(imgPatches.shape[1]):                
                imgPatchList.append(imgPatches[i, j, :, :])


        for i in range(gtPatches.shape[0]):
            for j in range(gtPatches.shape[1]):
                gtPatchList.append(gtPatches[i, j, :, :])
 

        posIdx = [idx for idx, x in enumerate(gtPatchList) if np.sum(x) > 100]
        negIdx = [x for x in range(len(gtPatchList)) if x not in posIdx]

        if len(negIdx) > len(posIdx):
            negIdx = sample(negIdx, len(posIdx))
        else:
            posIdx = sample(posIdx, len(negIdx))

                
        # get patches w positive data            
        posImg = [np.squeeze(imgPatchList[x], axis=0) for x in posIdx]
        posGt = [np.squeeze(gtPatchList[x], axis=0) for x in posIdx]
        
        # get patches w negative data
        negImg = [np.squeeze(imgPatchList[x], axis=0) for x in negIdx]
        negGt = [np.squeeze(gtPatchList[x], axis=0) for x in negIdx]
            
        # combine positive, negative patches for images & gt
        imgPatches = posImg + negImg
        gtPatches = posGt + negGt
        
        return imgPatches, gtPatches


    def _positivePatches(self, img, gt, patchSize):
        step = round(int(patchSize) * 0.15)
        #print(step)
        imgPatchList = []
        gtPatchList = []
        imgPatches = patchify(img, (patchSize, patchSize, 1), step=step)
        gtPatches = patchify(gt, (patchSize, patchSize, 1), step=step)

        for i in range(imgPatches.shape[0]):
            for j in range(imgPatches.shape[1]):
                imgPatchList.append(imgPatches[i, j, :, :])


        for i in range(gtPatches.shape[0]):
            for j in range(gtPatches.shape[1]):
                gtPatchList.append(gtPatches[i, j, :, :])


        posIdx = [idx for idx, x in enumerate(gtPatchList) if np.sum(x) > 50]

        # get patches w positive data
        posImg = [np.squeeze(imgPatchList[x], axis=0) for x in posIdx]
        posGt = [np.squeeze(gtPatchList[x], axis=0) for x in posIdx]

        return posImg, posGt



