
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

import cv2
from matplotlib import pyplot as plt
from matplotlib import pyplot

# Locally connected layer
class LocallyConnected2D(nn.Module):
    def calculate_spatial_output_shape(self, inputShape, kernelSize, dilation, padding, stride):
        return [np.floor(((inputShape[index]+2*padding[index]-dilation[index]*(kernelSize[index]-1)-1)/stride[index])+1).astype(int) for index in range(len(inputShape))]
    def __init__(self, inputShape, inChannels, outChannels, kernelSize, dilation, padding, stride):
        super(LocallyConnected2D, self).__init__()
        self.inputShape = inputShape
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        # Calculate desired output shape
        self.outputHeight, self.outputWidth = self.calculate_spatial_output_shape(self.inputShape, self.kernelSize, self.dilation, self.padding, self.stride)
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

# Transposed locally connected layer
class TransposedLocallyConnected2D(nn.Module):
    def calculate_transposed_padding(self, inputShape, outputShape, kernelSize, dilation, stride):
        return [np.ceil(((outputShape[index]-1)*stride[index]+dilation[index]*(kernelSize[index]-1)-inputShape[index]+1)/2).astype(int) for index in range(len(inputShape))]
    def __init__(self, inputShape, outputShape, inChannels, outChannels, kernelSize, dilation, stride):
        super(TransposedLocallyConnected2D, self).__init__()
        self.inputShape = inputShape
        self.outputShape = outputShape
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.dilation = dilation
        self.stride = stride
        # compute the padding for the transposed operation
        self.padding = self.calculate_transposed_padding(self.inputShape, self.outputShape, self.kernelSize, self.dilation, self.stride)
        # create a locally connected layer with the adapted padding
        self.transposedLC = LocallyConnected2D(self.inputShape, self.inChannels, self.outChannels, self.kernelSize, self.dilation, self.padding, self.stride)

    def forward(self, input):
        return self.transposedLC(input)

# get image and resize it
img = cv2.imread('cat.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (128,128))

plt.imshow(img)
plt.show()

input = torch.reshape(torch.from_numpy(img), (3,128,128))
input = input.view((1, *input.shape)).type(torch.FloatTensor)
print(input.shape)

inLayer = LocallyConnected2D(inputShape=[128,128], inChannels=3, outChannels=49, kernelSize=[5,5], dilation=[1,1], padding=[0,0], stride=[5,5])
outLayer = TransposedLocallyConnected2D(inputShape=[25,25], outputShape=[128,128], inChannels=49, outChannels=3, kernelSize=[5,5], dilation=[1,1], stride=[5,5])

ht = inLayer(input)
for i in range(0,10):
    fHt = ht[0,i,:,:]
    plt.matshow(fHt.detach().numpy())
    plt.show()

y = outLayer(ht)
for i in range(0,3):
    fy = y[0,i,:,:]
    plt.matshow(fy.detach().numpy())
    plt.show()
