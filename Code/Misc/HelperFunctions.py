"""
@file    HelperFunctions.py
@author  rohithjayarajan
@date 02/17/2019

Licensed under the
GNU General Public License v3.0
"""

import numpy as np
import cv2


class HelperFunctions:

    def ComputeSSD(self, Vector1, Vector2):
        return ((Vector1-Vector2)**2).sum()

    def DrawMatches(self, Image1, Image2, Kp1, Kp2):
        (hA, wA) = Image1.shape[:2]
        (hB, wB) = Image2.shape[:2]
        Draw = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        Draw[0:hA, 0:wA] = Image1
        Draw[0:hB, wA:] = Image2

        for IdY in range(0, len(Kp1)):
            ptA = (int(Kp1[IdY].pt[0]),
                   int(Kp1[IdY].pt[1]))
            ptB = (int(Kp2[IdY].pt[0]) +
                   wA, int(Kp2[IdY].pt[1]))
            cv2.line(Draw, ptA, ptB, (0, 0, 255), 1)

        self.ShowImage(Draw, 'Matches')

    def ShowImage(self, Image, ImgName):
        cv2.imshow(ImgName, Image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def ConvertTo3D(self, Vector2D):
        Vector3D = np.array([Vector2D[1], Vector2D[0], 1])
        return Vector3D

    def ConvertToHomogeneous(self, Point):
        return Point/float(Point[2])

    def SSDRansac(self, MatchingFeatures, H):
        SSDVal = np.zeros(len(MatchingFeatures))
        SSDiD = 0
        for Pair in MatchingFeatures:
            pi = self.ConvertTo3D(Pair[0])
            piDash = self.ConvertTo3D(Pair[1])
            Hpi = np.matmul(H, pi)
            Hpi = self.ConvertToHomogeneous(Hpi)
            SSDVal[SSDiD] = self.ComputeSSD(piDash, Hpi)
            SSDiD += 1

        return SSDVal
