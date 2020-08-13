#     _   __                      ______
#    / | / /__  __  ___________  / ____/___  ________
#   /  |/ / _ \/ / / / ___/ __ \/ /   / __ \/ ___/ _ \
#  / /|  /  __/ /_/ / /  / /_/ / /___/ /_/ / /  /  __/
# /_/ |_/\___/\__,_/_/   \____/\____/\____/_/   \___/

# NeuroCore is an PyTorch implementation of a Predictive Visual Network for robotics
# applications (tracking, recognition, manipulation,...)

# Author : Munch Quentin, 2020

# Pytorch library
import torch
import torch.nn as nn
from torch.nn import init
# custom layer
from Locally_Connected_Layer import *

# autoencoding layer for hiearchical prediction
class PredictiveLayer(nn.Module):
    def calculate_padding(self, inputShape, outputShape, kernelSize, dilation, stride):
        return [np.ceil(((outputShape[index]-1)*stride[index]+dilation[index]*(kernelSize[index]-1)-inputShape[index]+1)/2).astype(int) for index in range(len(inputShape))]
    def __init__(self, inputSize, hiddenSize, feedBackSize, inChannels, hiddenChannels, feedBackChannels, kernelSize, lateralKerneSize, padding, stride):
        super(PredictiveLayer, self).__init__()
        # layer attributs :
        # dimensional attributs
        self.inputSize = inputSize
        self.inputChannels = inChannels
        self.hiddenSize = hiddenSize
        self.hiddenChannels = hiddenChannels
        self.fbSize = feedBackSize
        self.fbChannels = feedBackChannels
        self.outputSize = inputSize
        self.outputChannels = inChannels
        # operation attributs
        self.kernelSize = kernelSize
        self.lateralKerneSize = lateralKerneSize
        self.padding = padding
        self.stride = stride
        # special operation
        self.recurrentLateralPadding = self.calculate_padding(inputShape=self.hiddenSize,
                                                              outputShape=self.hiddenSize,
                                                              kernelSize=self.lateralKerneSize,
                                                              dilation=[1,1],
                                                              stride=[1,1])
        # layer defition :
        # main input encoder
        self.inputEncoder = LocallyConnected2D(inputShape=self.inputSize,
                                               inChannels=self.inputChannels,
                                               outChannels=self.hiddenChannels,
                                               kernelSize=self.kernelSize,
                                               dilation=[1,1],
                                               padding=self.padding,
                                               stride=self.stride)
        # prediction error encoder
        self.errorEncoder = LocallyConnected2D(inputShape=self.inputSize,
                                               inChannels=self.inputChannels,
                                               outChannels=self.hiddenChannels,
                                               kernelSize=self.kernelSize,
                                               dilation=[1,1],
                                               padding=self.padding,
                                               stride=self.stride)
        # lateral / recurrent encoder
        self.lateralRecurrentEncoder = LocallyConnected2D(inputShape=self.hiddenSize,
                                                          inChannels=self.hiddenChannels,
                                                          outChannels=self.hiddenChannels,
                                                          kernelSize=self.lateralKerneSize,
                                                          dilation=[1,1],
                                                          padding=self.recurrentLateralPadding,
                                                          stride=[1,1])
        # upper input feedback encoder
        self.feedbackEncoder = TransposedLocallyConnected2D(inputShape=self.fbSize,
                                                            outputShape=self.hiddenSize,
                                                            inChannels=self.fbChannels,
                                                            outChannels=self.hiddenChannels,
                                                            kernelSize=self.kernelSize,
                                                            dilation=[1,1],
                                                            stride=self.stride)
        # prediction decoder
        self.outputDecoder = TransposedLocallyConnected2D(inputShape=self.hiddenSize,
                                                          outputShape=self.inputSize,
                                                          inChannels=self.hiddenChannels,
                                                          outChannels=self.inputChannels,
                                                          kernelSize=self.kernelSize,
                                                          dilation=[1,1],
                                                          stride=self.stride)
        # activation function
        self.activation = torch.nn.ReLU6()

    def forward(self, Xt, Et, lastHt, lastFbt):
        # encoding
        HXt = self.inputEncoder(Xt)
        HEt = self.errorEncoder(Et)
        HLRt = self.lateralRecurrentEncoder(lastHt)
        HFbt = self.feedbackEncoder(lastFbt)
        # activation
        Ht = self.activation(HXt+HEt+HLRt+HFbt)
        # decoding
        Yt = self.outputDecoder(Ht)
        return Yt, Ht
