"""
@file    ImageUtils.py
@author  rohithjayarajan
@date 02/22/2019

Licensed under the
GNU General Public License v3.0
"""

import numpy as np
import cv2
import random

debug = False


class ImageUtils:

    def PreProcess(self, InputImage, ResizeHeight, ResizeWidth):
        PreProcessedImage = cv2.resize(InputImage, dsize=(
            ResizeHeight, ResizeWidth), interpolation=cv2.INTER_CUBIC)
        PreProcessedImage = cv2.cvtColor(PreProcessedImage, cv2.COLOR_BGR2GRAY)
        return PreProcessedImage

    def ImageStandardization(self, InputImage):
        mu = np.mean(InputImage)
        StandardizedImage = (InputImage - mu)/float(255)
        return StandardizedImage

    # def RandomPatch(self, InputImage):
    #     rho = 32
    #     # print(InputImage.shape)
    #     RightGap = InputImage.shape[1] - 128 - 40
    #     DownGap = InputImage.shape[0] - 128 - 40

    #     X = random.randint(40, RightGap)
    #     Y = random.randint(40, DownGap)

    #     CropA = [(X, Y),
    #              (X + 128, Y),
    #              (X + 128, Y + 128),
    #              (X, Y + 128)]
    #     X1Perturb = random.randint(-rho, rho)
    #     Y1Perturb = random.randint(-rho, rho)
    #     X2Perturb = random.randint(-rho, rho)
    #     Y2Perturb = random.randint(-rho, rho)
    #     X3Perturb = random.randint(-rho, rho)
    #     Y3Perturb = random.randint(-rho, rho)
    #     X4Perturb = random.randint(-rho, rho)
    #     Y4Perturb = random.randint(-rho, rho)
    #     CropB = [(X + X1Perturb, Y + Y1Perturb),
    #              (X + 128 + X2Perturb, Y + Y2Perturb),
    #              (X + 128 + X3Perturb, Y + 128 + Y3Perturb),
    #              (X + X4Perturb, Y + 128 + Y4Perturb)]

    #     return np.array(CropA), np.array(CropB)

    # def CreateTrainingData(self, InputImage):

    #     PatchSize = 128
    #     CropA, CropB = self.RandomPatch(InputImage)
    #     PatchA = InputImage[CropA[0][1]:CropA[0][1] +
    #                         PatchSize, CropA[0][0]:CropA[0][0] + PatchSize]
    #     H_AB = cv2.getPerspectiveTransform(
    #         np.float32(CropA), np.float32(CropB))
    #     H_BA = cv2.getPerspectiveTransform(
    #         np.float32(CropB), np.float32(CropA))
    #     WarpedImage = cv2.warpPerspective(InputImage, H_BA, (320, 240),
    #                                       flags=cv2.INTER_CUBIC)
    #     PatchB = WarpedImage[CropA[0][1]:CropA[0][1] +
    #                          PatchSize, CropA[0][0]:CropA[0][0] + PatchSize]
    #     # print(PatchA.shape)
    #     # print(PatchB.shape)
    #     Patches = np.stack((PatchA, PatchB), axis=-1)
    #     H4PtTruth = (CropB - CropA).reshape(-1)

    #     # cv2.imshow('patch1', PatchA)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()

    #     # cv2.imshow('patch2', PatchB)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()

    #     return Patches, H4PtTruth

    def RandomCrop(self, InputImage):
        TopGap = 40
        LeftGap = 40
        RightGap = InputImage.shape[1] - 128 - TopGap
        DownGap = InputImage.shape[0] - 128 - LeftGap
        # print(InputImage.shape)

        PatchY = random.randint(TopGap, DownGap+1)
        PatchX = random.randint(LeftGap, RightGap+1)

        Patch = InputImage[PatchY:PatchY+128, PatchX:PatchX+128]

        return Patch, PatchX, PatchY

    def CreateTrainingData(self, InputImage):
        rho = 32
        PatchA, PatchAX, PatchAY = self.RandomCrop(InputImage)
        # print(PatchA.shape)

        PatchAX1 = PatchAX
        PatchAY1 = PatchAY

        PatchAX2 = PatchAX + PatchA.shape[1]
        PatchAY2 = PatchAY

        PatchAX3 = PatchAX + PatchA.shape[1]
        PatchAY3 = PatchAY + PatchA.shape[0]

        PatchAX4 = PatchAX
        PatchAY4 = PatchAY + PatchA.shape[0]

        PatchBX1 = PatchAX + random.randint(-rho, rho)
        PatchBY1 = PatchAY + random.randint(-rho, rho)

        PatchBX2 = PatchAX + PatchA.shape[1] + random.randint(-rho, rho)
        PatchBY2 = PatchAY + random.randint(-rho, rho)

        PatchBX3 = PatchAX + PatchA.shape[1] + random.randint(-rho, rho)
        PatchBY3 = PatchAY + PatchA.shape[0] + random.randint(-rho, rho)

        PatchBX4 = PatchAX + random.randint(-rho, rho)
        PatchBY4 = PatchAY + PatchA.shape[0] + random.randint(-rho, rho)

        H4PtTruth = np.array([PatchBX1 - PatchAX1,
                              PatchBX2 - PatchAX2,
                              PatchBX3 - PatchAX3,
                              PatchBX4 - PatchAX4,
                              PatchBY1 - PatchAY1,
                              PatchBY2 - PatchAY2,
                              PatchBY3 - PatchAY3,
                              PatchBY4 - PatchAY4])

        if debug:

            cv2.line(InputImage, (PatchAX, PatchAY),
                     (PatchAX+128, PatchAY), (255, 0, 0), 5)
            cv2.imshow('plotter1', InputImage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.line(InputImage, (PatchBX1, PatchBY1),
                     (PatchBX2, PatchBY2), (0, 255, 0), 5)
            cv2.imshow('plotter1', InputImage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.line(InputImage, (PatchAX+128, PatchAY),
                     (PatchAX+128, PatchAY+128), (255, 0, 0), 5)
            cv2.imshow('plotter1', InputImage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.line(InputImage, (PatchBX2, PatchBY2),
                     (PatchBX3, PatchBY3), (0, 255, 0), 5)
            cv2.imshow('plotter1', InputImage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.line(InputImage, (PatchAX+128, PatchAY+128),
                     (PatchAX, PatchAY+128), (255, 0, 0), 5)
            cv2.imshow('plotter1', InputImage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.line(InputImage, (PatchBX3, PatchBY3),
                     (PatchBX4, PatchBY4), (0, 255, 0), 5)
            cv2.imshow('plotter1', InputImage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.line(InputImage, (PatchAX, PatchAY+128),
                     (PatchAX, PatchAY), (255, 0, 0), 5)
            cv2.imshow('plotter1', InputImage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.line(InputImage, (PatchBX4, PatchBY4),
                     (PatchBX1, PatchBY1), (0, 255, 0), 5)
            cv2.imshow('plotter2', InputImage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Points1 = np.float32([[PatchAX, PatchAY], [
        #     PatchAX+128, PatchAY], [PatchAX+128, PatchAY+128], [PatchAX, PatchAY+128]])
        Points1 = np.float32([[PatchAX1, PatchAY1], [PatchAX2, PatchAY2], [
            PatchAX3, PatchAY3], [PatchAX4, PatchAY4]])
        Points2 = np.float32([[PatchBX1, PatchBY1], [PatchBX2, PatchBY2], [
            PatchBX3, PatchBY3], [PatchBX4, PatchBY4]])

        H_AB = cv2.getPerspectiveTransform(Points1, Points2)
        H_BA = cv2.getPerspectiveTransform(Points2, Points1)

        WarpedImage = cv2.warpPerspective(
            InputImage, H_BA, (InputImage.shape[1], InputImage.shape[0]))

        PatchB = WarpedImage[PatchAY:PatchAY+128, PatchAX:PatchAX+128]

        if debug:
            cv2.imshow('patch1', PatchA)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.imshow('patch2', PatchB)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.imshow('IMAGEREV', cv2.warpPerspective(
                WarpedImage, H_AB, (InputImage.shape[1], InputImage.shape[0])))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.imshow('patch2rev', cv2.warpPerspective(
                PatchB, H_AB, (PatchB.shape[1]+100, PatchB.shape[0]+100)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        Patches = np.dstack((PatchA, PatchB))

        return Patches, H4PtTruth
