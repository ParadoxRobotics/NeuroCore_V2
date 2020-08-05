#    _   __                      ______
#   / | / /__  __  ___________  / ____/___  ________
#  /  |/ / _ \/ / / / ___/ __ \/ /   / __ \/ ___/ _ \
# / /|  /  __/ /_/ / /  / /_/ / /___/ /_/ / /  /  __/
#/_/ |_/\___/\__,_/_/   \____/\____/\____/_/   \___/

# NeuroCore is an PyTorch implementation of a Predictive Visual Model for robotics
# applications (tracking, recognition, manipulation,...)

# Author : Munch Quentin, 2020

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init

# Locally connected layer
class LocallyConnected2D(nn.Module):
    def calculate_spatial_output_shape(self, inputShape, kernelSize, dilation, padding, stride):
        return [np.floor(((inputShape[index]+2*padding[index]-dilation[index]*(kernelSize[index]-1)-1)/stride[index])+1).astype(int) for index in range(len(inputShape))]
    def __init__(self, inputShape, inChannels, outChannels, kernelSize, dilation, padding, stride):
        super().__init__()
        self.inputShape = inputShape
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        # Calculate desired output shape
        self.outputHeight, self.outputWidth = self.calculate_spatial_output_shape(self.inputShape, self.kernelSize, self.dilation, self.padding, self.stride)
        print(self.outputHeight, self.outputWidth)
        self.weightTensorDepth = self.inChannels * self.kernelSize[0] * self.kernelSize[1]
        self.spatialBlocksSize = self.outputHeight * self.outputWidth
        # init weight and bias
        self.weights = nn.Parameter(torch.empty((1, self.weightTensorDepth, self.spatialBlocksSize, self.outChannels),requires_grad=True, dtype=torch.float))
        torch.nn.init.xavier_uniform_(self.weights)
        self.bias = nn.Parameter(torch.empty((1, outChannels, self.outputHeight, self.outputWidth),requires_grad=True, dtype=torch.float))
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, input):
        # Perform Vol2Col/Im2Col operation on the input feature given kernel, stride, padding and dilation size
        inputUnfold = torch.nn.functional.unfold(input, self.kernelSize, dilation=self.dilation, padding=self.padding, stride=self.stride)
        # Apply the weight to the unfolded image
        localOpUnfold = (inputUnfold.view((*inputUnfold.shape, 1)) * self.weights)
        return localOpUnfold.sum(dim=1).transpose(2, 1).reshape((-1, self.outChannels, self.outputHeight, self.outputWidth)) + self.bias

# Exemple test :
# distributed locally connected layer where there is no overlaping over the receptive field
inLayer = LocallyConnected2D(inputShape=[128,128], inChannels=3, outChannels=49, kernelSize=[5,5], dilation=[1,1], padding=[0,0], stride=[5,5])
errorLayer = LocallyConnected2D(inputShape=[128,128], inChannels=3, outChannels=49, kernelSize=[5,5], dilation=[1,1], padding=[0,0], stride=[5,5])
# locally connected lateral/recurrent connection -> each neuron see the prevous state of their neighbor (RF = 5X5)
latRLayer = LocallyConnected2D(inputShape=[25,25], inChannels=49, outChannels=49, kernelSize=[5,5], dilation=[1,1], padding=[2,2], stride=[1,1])
