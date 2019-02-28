#!/usr/bin/evn python

"""
@file    Wrapper.py
@author  rohithjayarajan
@date 02/22/2019

Licensed under the
GNU General Public License v3.0
"""

import numpy as np
import tensorflow as tf
import cv2
import argparse
from PIL import Image
import glob
from skimage.feature import peak_local_max
import math
import random
import matplotlib.pyplot as plt
from Misc.HelperFunctions import HelperFunctions
from Misc.ImageUtils import ImageUtils
from Network.Network import DeepNetwork
from tqdm import tqdm


class Stitcher:
    """
    Read a set of images for Panorama stitching
    """

    def __init__(self, BasePath, ModelPath, NumFeatures):
        self.BasePath = BasePath
        self.ModelPath = ModelPath
        InputImageList = []
        for filename in sorted(glob.glob(self.BasePath + '/*.jpg')):
            ImageTemp = cv2.imread(filename)
            InputImageList.append(ImageTemp)
        self.NumFeatures = NumFeatures
        self.Images = np.array(InputImageList)
        self.NumImages = len(InputImageList)
        self.HelperFunctions = HelperFunctions()
        self.Model = DeepNetwork()
        self.ImageUtils = ImageUtils()
        self.ImageSize = InputImageList[0].shape
        self.ImgPH = tf.placeholder('float', shape=(1, 128, 128, 2))

    """
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""

    def ExtractHomographyFromH4Pt(self, H4PtPred):

        pts1 = np.float32([[0, 0], [self.ImageSize[1], 0], [
                          self.ImageSize[1], self.ImageSize[0]], [0, self.ImageSize[0]]])

        pts2 = np.float32([[0+H4PtPred[0][0], 0+H4PtPred[0][4]], [self.ImageSize[1]+H4PtPred[0][1], 0+H4PtPred[0][5]], [
                          self.ImageSize[1]+H4PtPred[0][2], self.ImageSize[0]+H4PtPred[0][6]], [0+H4PtPred[0][3], self.ImageSize[0]+H4PtPred[0][7]]])
        HPred = cv2.getPerspectiveTransform(pts1, pts2)
        return HPred

    def EstimateHomographySupervised(self, Image1, Image2):
        # Setup Saver
        H4PtPred = self.Model.HomographyNet(self.ImgPH, False)
        Saver = tf.train.Saver()

        with tf.Session() as sess:
            Saver.restore(sess, self.ModelPath)
            print('Number of parameters in this model are %d ' % np.sum(
                [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

            Image1 = np.float32(self.ImageUtils.PreProcess(Image1, 128, 128))
            Image1 = self.ImageUtils.ImageStandardization(Image1)
            Image2 = np.float32(self.ImageUtils.PreProcess(Image2, 128, 128))
            Image2 = self.ImageUtils.ImageStandardization(Image2)

            Images = np.dstack((Image1, Image2))
            I1Batch = []
            I1Batch.append(Images)
            FeedDict = {self.ImgPH: I1Batch}
            H4Pt = sess.run(H4PtPred, FeedDict)
            print("H4pt: {}".format(H4Pt))
            H = self.ExtractHomographyFromH4Pt(H4Pt)
            Hinv = np.linalg.inv(H)

        return H, Hinv

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""

    def RemoveBlackBoundary(self, ImageIn):
        gray = cv2.cvtColor(ImageIn, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        ImageOut = ImageIn[y:y+h, x:x+w]
        return ImageOut

    def Warping(self, Img, Homography, NextShape):
        nH, nW, _ = Img.shape
        Borders = np.array([[0, nW, nW, 0], [0, 0, nH, nH], [1, 1, 1, 1]])
        BordersNew = np.dot(Homography, Borders)
        Ymin = min(BordersNew[1]/BordersNew[2])
        Xmin = min(BordersNew[0]/BordersNew[2])
        Ymax = max(BordersNew[1]/BordersNew[2])
        Xmax = max(BordersNew[0]/BordersNew[2])
        if Ymin < 0:
            MatChange = np.array(
                [[1, 0, -1 * Xmin], [0, 1, -1 * Ymin], [0, 0, 1]])
            Hnew = np.dot(MatChange, Homography)
            h = int(round(Ymax - Ymin)) + NextShape[0]
        else:
            MatChange = np.array(
                [[1, 0, -1 * Xmin], [0, 1, Ymin], [0, 0, 1]])
            Hnew = np.dot(MatChange, Homography)
            h = int(round(Ymax + Ymin)) + NextShape[0]
        w = int(round(Xmax - Xmin)) + NextShape[1]
        sz = (w, h)
        PanoHolder = cv2.warpPerspective(Img, Hnew, dsize=sz)
        return PanoHolder, int(Xmin), int(Ymin)

    def Blender(self):
        Pano = self.Images[0]
        for NextImage in self.Images[1:2]:
            H, Hinv = self.EstimateHomographySupervised(Pano, NextImage)
            PanoHolder, oX, oY = self.Warping(Pano, H, NextImage.shape)
            self.HelperFunctions.ShowImage(PanoHolder, 'PanoHolder')
            oX = abs(oX)
            oY = abs(oY)
            for IdY in range(oY, NextImage.shape[0]+oY):
                for IdX in range(oX, NextImage.shape[1]+oX):
                    y = IdY - oY
                    x = IdX - oX
                    PanoHolder[IdY, IdX, :] = NextImage[y, x, :]
            # Pano = self.RemoveBlackBoundary(PanoHolder)
        PanoResize = cv2.resize(Pano, (1280, 1024))
        self.HelperFunctions.ShowImage(PanoResize, 'PanoResize')
        PanoResize = cv2.GaussianBlur(PanoResize, (5, 5), 1.2)
        return PanoResize


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='/home/rohith/CMSC733/git/Homography-Net/Data/Train2/Set1',
                        help='Base path of images, Default:/home/rohith/CMSC733/git/Homography-Net/Data/Train2/Set1')
    Parser.add_argument('--ModelPath', dest='ModelPath', default='/home/rohith/CMSC733/git/Homography-Net/Checkpoints/99model.ckpt',
                        help='Path to load latest model from, Default:/home/rohith/CMSC733/git/Homography-Net/Checkpoints/49model.ckpt')
    Parser.add_argument('--NumFeatures', default='700',
                        help='Number of best features to extract from each image, Default:100')
    Args = Parser.parse_args()
    NumFeatures = int(Args.NumFeatures)
    BasePath = Args.BasePath
    ModelPath = Args.ModelPath

    myStitcher = Stitcher(BasePath, ModelPath, NumFeatures)
    Pano = myStitcher.Blender()


if __name__ == '__main__':
    main()
