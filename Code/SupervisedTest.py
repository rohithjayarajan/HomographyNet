#!/usr/bin/evn python

"""
@file    SupervisedTest.py
@author  rohitkrishna nambiar
@date 02/22/2019

Licensed under the
GNU General Public License v3.0
"""

# Code starts here:

import numpy as np
import cv2
import tensorflow as tf
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import DeepNetwork
from Misc.MiscUtils import *
import time
import argparse
import shutil
from StringIO import StringIO
import string
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *
# Add any python libraries here


def GetPatchImage(img):
    rho = 32
    patchSize = 128
    padding = 35  # Make sure padding is more than rho

    # Read input image
    height, width = img.shape  # row, col

    randX = np.random.random_integers(padding, width - patchSize - padding)
    randY = np.random.random_integers(padding, height - patchSize - padding)

    # Step:1 - Get image patch
    patch = img[randY:randY+patchSize, randX:randX+patchSize]

    # Step:2 - Perturn the 4 corners (Clockwise)
    pertX = [randX+np.random.random_integers(-rho, rho),
             randX+patchSize+np.random.random_integers(-rho, rho),
             randX+patchSize+np.random.random_integers(-rho, rho),
             randX+np.random.random_integers(-rho, rho)]

    pertY = [randY+np.random.random_integers(-rho, rho),
             randY+np.random.random_integers(-rho, rho),
             randY+patchSize+np.random.random_integers(-rho, rho),
             randY+patchSize+np.random.random_integers(-rho, rho)]

    src_pts = np.array([[randX, randY], [randX+patchSize, randY], [randX +
                                                                   patchSize, randY+patchSize], [randX, randY+patchSize]], np.float32)
    dst_pts = np.array([[pertX[0], pertY[0]], [pertX[1], pertY[1]], [
                       pertX[2], pertY[2]], [pertX[3], pertY[3]]], np.float32)

    # Step:3 - Computing Homography
    Hab = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Hba = np.linalg.inv(Hab)

    # Step:4 - Wrap image and get patch
    Ib = cv2.warpPerspective(img, Hba, (width, height))
    patch2 = Ib[randY:randY+patchSize, randX:randX+patchSize]

    # Normalizing image and stack them
    m1 = np.mean(patch)
    m2 = np.mean(patch2)
    patch = (patch - m1)/255.0
    patch2 = (patch2 - m2)/255.0
    stacked = np.dstack((patch, patch2))

    du0 = dst_pts[0][0] - src_pts[0][0]
    du1 = dst_pts[1][0] - src_pts[1][0]
    du2 = dst_pts[2][0] - src_pts[2][0]
    du3 = dst_pts[3][0] - src_pts[3][0]
    dv0 = dst_pts[0][1] - src_pts[0][1]
    dv1 = dst_pts[1][1] - src_pts[1][1]
    dv2 = dst_pts[2][1] - src_pts[2][1]
    dv3 = dst_pts[3][1] - src_pts[3][1]

    error = np.array([du0, du1, du2, du3, dv0, dv1, dv2, dv3])

    return stacked, error, patch, patch2, src_pts, dst_pts


def Predict(ImgPH, Image, ModelPath):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    Image is the combined gray scale image
    ModelPath - Path to load trained model from 
    """
    ImageSize = 128
    net = DeepNetwork()

    # Predict output with forward pass, MiniBatchSize for Test is 1
    H4Pt = net.HomographyNet(ImgPH, isTrain=False)

    # Setup Saver
    Saver = tf.train.Saver()

    PredH4Pt = 0

    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        print('Number of parameters in this model are %d ' % np.sum(
            [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

        FeedDict = {ImgPH: Image}
        PredH4Pt = sess.run(H4Pt, FeedDict)
        print(PredH4Pt)

    return PredH4Pt[0]


def main():
        # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='/home/rohith/CMSC733/git/Homography-Net/Checkpoints/99model.ckpt',
                        help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--ImagePath', dest='ImagePath', default='/home/rohith/CMSC733/git/Homography-Net/Data/Train2/Set1/1.jpg',
                        help='Path to load images to create Panaroma, Default:BasePath')

    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    ImagePath = Args.ImagePath

    # Output blended Image
    blendImg = None
    ImageSize = (128, 128)

    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder('float', shape=(1, ImageSize[0], ImageSize[1], 2))

    # Read input image
    image = cv2.imread(ImagePath)
    image = cv2.resize(image, (320, 240))
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    stacked, error, patch, patch2, src_pts, dst_pts = GetPatchImage(image)
    stacked = stacked.reshape(1, 128, 128, 2)

    # for i in range(3):
    # 	cv2.line(image, tuple(src_pts[i]), tuple(src_pts[i+1]), (0, 0, 255), 2)

    # cv2.line(image, tuple(src_pts[0]), tuple(src_pts[3]), (0, 0, 255), 2)

    for i in range(3):
        cv2.line(image, tuple(dst_pts[i]), tuple(dst_pts[i+1]), (0, 255, 0), 2)

    cv2.line(image, tuple(dst_pts[0]), tuple(dst_pts[3]), (0, 255, 0), 2)

    """
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""
    PredH4Pt = Predict(ImgPH, stacked, ModelPath)

    dst_pts_pred = np.array([[src_pts[0][0] + PredH4Pt[0], src_pts[0][1] + PredH4Pt[4]],
                             [src_pts[1][0] + PredH4Pt[1],
                                 src_pts[1][1] + PredH4Pt[5]],
                             [src_pts[2][0] + PredH4Pt[2],
                                 src_pts[2][1] + PredH4Pt[6]],
                             [src_pts[3][0] + PredH4Pt[3], src_pts[3][1] + PredH4Pt[7]]], np.float32)

    for i in range(3):
        cv2.line(image, tuple(dst_pts_pred[i]), tuple(
            dst_pts_pred[i+1]), (255, 0, 0), 2)

    cv2.line(image, tuple(dst_pts_pred[0]),
             tuple(dst_pts_pred[3]), (255, 0, 0), 2)

    # cv2.putText(image,'Source',(10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1,cv2.LINE_AA)
    # cv2.putText(image,'Perturb GT',(10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1,cv2.LINE_AA)
    # cv2.putText(image,'Predicted GT',(10,140), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0),1,cv2.LINE_AA)

    # Prints
    print("Source points: {}".format(src_pts))
    print("Dest points: {}".format(dst_pts))
    print("Dest points Predicted: {}".format(dst_pts_pred))

    print("Prediction H4Pt: {}".format(PredH4Pt))
    print("Error ground truth: {}".format(error))
    print("Total pixel error: {}".format(np.sum(np.abs(PredH4Pt-error))))

    # Warp image
    Hab = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Hba = np.linalg.inv(Hab)

    height, width, _ = image.shape
    Ib = cv2.warpPerspective(image, Hab, (width, height))

    cv2.imshow("i1", patch)
    cv2.moveWindow("i1", 200, 200)

    cv2.imshow("i2", patch2)
    cv2.moveWindow("i2", 400, 400)

    cv2.imshow("i3", Ib)
    cv2.moveWindow("i3", 600, 200)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
