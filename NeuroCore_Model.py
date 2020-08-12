#     _   __                      ______
#    / | / /__  __  ___________  / ____/___  ________
#   /  |/ / _ \/ / / / ___/ __ \/ /   / __ \/ ___/ _ \
#  / /|  /  __/ /_/ / /  / /_/ / /___/ /_/ / /  /  __/
# /_/ |_/\___/\__,_/_/   \____/\____/\____/_/   \___/

# NeuroCore is an PyTorch implementation of a Predictive Visual Network for robotics
# applications (tracking, recognition, manipulation,...)

# Author : Munch Quentin, 2020

# general library
import numpy as np
import cv2
import matplotlib.pyplot as plt
# Pytorch library
import torch
import torch.nn as nn
from torch.nn import init
# custom layer
from Predictive_Layer import *

class NeuroCore_A(nn.Module):
    def __init__(self):
        super(NeuroCore_A, self).__init__()
        # predictive visual hierarchy
        self.LPU_1 = PredictiveLayer(inputSize=[128,128],
                                     hiddenSize=[64,64],
                                     feedBackSize=[32,32],
                                     inChannels=3,
                                     hiddenChannels=49,
                                     feedBackChannels=49,
                                     kernelSize=[5,5],
                                     lateralKerneSize=[5,5],
                                     padding=[2,2],
                                     stride=[2,2])

        self.LPU_2 = PredictiveLayer(inputSize=[64,64],
                                     hiddenSize=[32,32],
                                     feedBackSize=[16,16],
                                     inChannels=49,
                                     hiddenChannels=49,
                                     feedBackChannels=49,
                                     kernelSize=[5,5],
                                     lateralKerneSize=[5,5],
                                     padding=[2,2],
                                     stride=[2,2])

        self.LPU_3 = PredictiveLayer(inputSize=[32,32],
                                     hiddenSize=[16,16],
                                     feedBackSize=[8,8],
                                     inChannels=49,
                                     hiddenChannels=49,
                                     feedBackChannels=49,
                                     kernelSize=[5,5],
                                     lateralKerneSize=[5,5],
                                     padding=[2,2],
                                     stride=[2,2])

        self.LPU_4 = PredictiveLayer(inputSize=[16,16],
                                     hiddenSize=[8,8],
                                     feedBackSize=[1,1],
                                     inChannels=49,
                                     hiddenChannels=49,
                                     feedBackChannels=1,
                                     kernelSize=[5,5],
                                     lateralKerneSize=[5,5],
                                     padding=[2,2],
                                     stride=[2,2])

    # Inference step
    def forward(self, Xt, Et_1, Et_2, Et_3, Et_4, lastHt_1, lastHt_2, lastHt_3, lastHt_4, upCmd):
        Yt_1, Ht_1 = self.LPU_1(Xt, Et_1, lastHt_1, lastHt_2)
        Yt_2, Ht_2 = self.LPU_2(Ht_1, Et_2, lastHt_2, lastHt_3)
        Yt_3, Ht_3 = self.LPU_3(Ht_2, Et_3, lastHt_3, lastHt_4)
        Yt_4, Ht_4 = self.LPU_4(Ht_3, Et_4, lastHt_4, upCmd)
        return Yt_1, Yt_2, Yt_3, Yt_4, Ht_1, Ht_2, Ht_3, Ht_4


class NeuroCore_B(nn.Module):
    def __init__(self):
        super(NeuroCore_B, self).__init__()
        # predictive visual hierarchy
        self.LPU_1 = PredictiveLayer(inputSize=[128,128],
                                     hiddenSize=[25,25],
                                     feedBackSize=[12,12],
                                     inChannels=3,
                                     hiddenChannels=49,
                                     feedBackChannels=49,
                                     kernelSize=[5,5],
                                     lateralKerneSize=[5,5],
                                     padding=[0,0],
                                     stride=[5,5])

        self.LPU_2 = PredictiveLayer(inputSize=[25,25],
                                     hiddenSize=[12,12],
                                     feedBackSize=[6,6],
                                     inChannels=49,
                                     hiddenChannels=49,
                                     feedBackChannels=49,
                                     kernelSize=[2,2],
                                     lateralKerneSize=[5,5],
                                     padding=[0,0],
                                     stride=[2,2])

        self.LPU_3 = PredictiveLayer(inputSize=[12,12],
                                     hiddenSize=[6,6],
                                     feedBackSize=[3,3],
                                     inChannels=49,
                                     hiddenChannels=49,
                                     feedBackChannels=49,
                                     kernelSize=[2,2],
                                     lateralKerneSize=[5,5],
                                     padding=[0,0],
                                     stride=[2,2])

        self.LPU_4 = PredictiveLayer(inputSize=[6,6],
                                     hiddenSize=[3,3],
                                     feedBackSize=[1,1],
                                     inChannels=49,
                                     hiddenChannels=49,
                                     feedBackChannels=49,
                                     kernelSize=[2,2],
                                     lateralKerneSize=[1,1],
                                     padding=[0,0],
                                     stride=[2,2])

        self.LPU_5 = PredictiveLayer(inputSize=[3,3],
                                     hiddenSize=[1,1],
                                     feedBackSize=[1,1],
                                     inChannels=49,
                                     hiddenChannels=49,
                                     feedBackChannels=1,
                                     kernelSize=[2,2],
                                     lateralKerneSize=[1,1],
                                     padding=[0,0],
                                     stride=[2,2])

    # Inference step
    def forward(self, Xt, Et_1, Et_2, Et_3, Et_4 , Et_5, lastHt_1, lastHt_2, lastHt_3, lastHt_4, lastHt_5, upCmd):
        Yt_1, Ht_1 = self.LPU_1(Xt, Et_1, lastHt_1, lastHt_2)
        Yt_2, Ht_2 = self.LPU_2(Ht_1, Et_2, lastHt_2, lastHt_3)
        Yt_3, Ht_3 = self.LPU_3(Ht_2, Et_3, lastHt_3, lastHt_4)
        Yt_4, Ht_4 = self.LPU_4(Ht_3, Et_4, lastHt_4, lastHt_5)
        Yt_5, Ht_5 = self.LPU_5(Ht_4, Et_5, lastHt_5, upCmd)
        return Yt_1, Yt_2, Yt_3, Yt_4, Yt_5, Ht_1, Ht_2, Ht_3, Ht_4, Ht_5
