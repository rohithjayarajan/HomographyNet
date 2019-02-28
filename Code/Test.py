#!/usr/bin/env python

"""
@file    Test.py
@author  rohithjayarajan
@date 02/22/2019

Licensed under the
GNU General Public License v3.0
"""

import tensorflow as tf
import cv2
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import HomographyModel
import numpy as np
import time
import argparse
import shutil
from StringIO import StringIO
import string
import math as m
from tqdm import tqdm
from Misc.ImageUtils import ImageUtils
from Misc.MiscUtils import *
from Misc.DataUtils import *


# Don't generate pyc codes
sys.dont_write_bytecode = True


class Test:
    def __init__(self):
        self.ImageUtils = ImageUtils()

    def SetupAll(self, BasePath):
        """
        Inputs: 
        BasePath - Path to images
        Outputs:
        ImageSize - Size of the Image
        DataPath - Paths of all images where testing will be run on
        """
        # Image Input Shape
        ImageSize = [32, 32, 3]
        DataPath = []
        NumImages = len(glob.glob(BasePath+'*.jpg'))
        SkipFactor = 1
        for count in range(1, NumImages+1, SkipFactor):
            DataPath.append(BasePath + str(count) + '.jpg')

        return ImageSize, DataPath

    def ReadImages(self, ImageSize, DataPath):
        """
        Inputs: 
        ImageSize - Size of the Image
        DataPath - Paths of all images where testing will be run on
        Outputs:
        I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
        I1 - Original I1 image for visualization purposes only
        """

        ImageName = DataPath

        I1 = cv2.imread(ImageName)

        if(I1 is None):
            # OpenCV returns empty list if image is not read!
            print('ERROR: Image I1 cannot be read')
            sys.exit()

        ##########################################################################
        # Add any standardization or cropping/resizing if used in Training here!
        ##########################################################################

        Im = self.ImageUtils.PreProcess(I1, 640, 480)
        I1S, H4PtTruth1 = self.ImageUtils.CreateTrainingData(Im, 256, 256, 64)
        I2S, H4PtTruth2 = self.ImageUtils.CreateTrainingData(Im, 256, 256, 64)

        I1Combined = np.expand_dims(I1S, axis=0)

        return I1Combined, I1

    def TestOperation(self, ImgPH, ImageSize, ModelPath, DataPath, LabelsPathPred):
        """
        Inputs: 
        ImgPH is the Input Image placeholder
        ImageSize is the size of the image
        ModelPath - Path to load trained model from
        DataPath - Paths of all images where testing will be run on
        LabelsPathPred - Path to save predictions
        Outputs:
        Predictions written to ./TxtFiles/PredOut.txt
        """
        Length = ImageSize[0]
        # Predict output with forward pass, MiniBatchSize for Test is 1
        _, prSoftMaxS = HomographyModel(ImgPH, ImageSize, 1)

        # Setup Saver
        Saver = tf.train.Saver()

        with tf.Session() as sess:
            Saver.restore(sess, ModelPath)
            print('Number of parameters in this model are %d ' % np.sum(
                [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

            OutSaveT = open(LabelsPathPred, 'w')

            for count in tqdm(range(np.size(DataPath))):
                DataPathNow = DataPath[count]
                Img, ImgOrg = self.ReadImages(ImageSize, DataPathNow)
                FeedDict = {ImgPH: Img}
                PredT = np.argmax(sess.run(prSoftMaxS, FeedDict))

                OutSaveT.write(str(PredT)+'\n')

            OutSaveT.close()

    def ReadLabels(self, LabelsPathTest, LabelsPathPred):
        if(not (os.path.isfile(LabelsPathTest))):
            print('ERROR: Test Labels do not exist in '+LabelsPathTest)
            sys.exit()
        else:
            LabelTest = open(LabelsPathTest, 'r')
            LabelTest = LabelTest.read()
            LabelTest = map(float, LabelTest.split())

        if(not (os.path.isfile(LabelsPathPred))):
            print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
            sys.exit()
        else:
            LabelPred = open(LabelsPathPred, 'r')
            LabelPred = LabelPred.read()
            LabelPred = map(float, LabelPred.split())

        return LabelTest, LabelPred


def main():
    """
    Inputs: 
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='/home/chahatdeep/Downloads/Checkpoints/144model.ckpt',
                        help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--BasePath', dest='BasePath', default='/home/rohith/CMSC733/git/Homography-Net/Data/Test/',
                        help='Path to load images from, Default:BasePath')
    Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./TxtFiles/LabelsTest.txt',
                        help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    LabelsPath = Args.LabelsPath

    # Setup all needed parameters including file reading
    ImageSize, DataPath = SetupAll(BasePath)

    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder('float', shape=(1, ImageSize[0], ImageSize[1], 3))
    LabelsPathPred = './TxtFiles/PredOut.txt'  # Path to save predicted labels

    TestOperation(ImgPH, ImageSize, ModelPath, DataPath, LabelsPathPred)

    # Plot Confusion Matrix
    LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
    ConfusionMatrix(LabelsTrue, LabelsPred)


if __name__ == '__main__':
    main()
