
#    _   __                      ______
#   / | / /__  __  ___________  / ____/___  ________
#  /  |/ / _ \/ / / / ___/ __ \/ /   / __ \/ ___/ _ \
# / /|  /  __/ /_/ / /  / /_/ / /___/ /_/ / /  /  __/
#/_/ |_/\___/\__,_/_/   \____/\____/\____/_/   \___/

# NeuroCore is an PyTorch implementation of a Predictive Visual Model for robotics
# applications (tracking, recognition, manipulation,...)

# Author : Munch Quentin, 2020

"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

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
        self.weightTensorDepth = self.inChannels * self.kernelSize[0] * self.kernelSize[1]
        self.spatialBlocksSize = self.outputHeight * self.outputWidth
        # init weight and bias
        self.weights = nn.Parameter(torch.empty((1, self.weightTensorDepth, self.spatialBlocksSize, self.outChannels),requires_grad=True, dtype=torch.float))
        torch.nn.init.xavier_uniform_(self.weights)
        self.bias = nn.Parameter(torch.empty((1, outChannels, self.outputHeight, self.outputWidth),requires_grad=True, dtype=torch.float))
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, input):
        # Perform Vol2Col operation on the input feature given kernel, stride, padding and dilation size
        inputUnfold = torch.nn.functional.unfold(input, self.kernelSize, dilation=self.dilation, padding=self.padding, stride=self.stride)
        # Apply the weight to the unfolded image
        localOpUnfold = (inputUnfold.view((*inputUnfold.shape, 1)) * self.weights)
        return localOpUnfold.sum(dim=1).transpose(2, 1).reshape((-1, self.outChannels, self.outputHeight, self.outputWidth)) + self.bias

# Transposed locally connected Layer
class TransposedLocallyConnected2D(nn.Module):
    def calculate_spatial_transposed_output_shape(self, inputShape, kernelSize, dilation, inputPadding, outPadding, stride):
        return [np.floor((inputShape[index]-1)*stride[index]-2*inputPadding[index]+dilation[index]*kernelSize[index]-1+outPadding[index]+1).astype(int) for index in range(len(inputShape))]
    def __init__(self, inputShape, inChannels, outChannels, kernelSize, dilation, inputPadding, outPadding, stride):
        super().__init__()
        self.inputShape = inputShape
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.dilation = dilation
        self.inputPadding = inputPadding
        self.outPadding = outPadding
        self.stride = stride
        # calculate desired output shape
        self.outputHeight, self.outputWidth = self.calculate_spatial_transposed_output_shape(self.inputShape, self.kernelSize, self.dilation, self.inputPadding, self.outPadding, self.stride)
        self.weightTensorDepth = self.outChannels * self.kernelSize[0] * self.kernelSize[1]
        self.spatialBlocksSize = self.outputHeight * self.outputWidth
        # init weight and bias
        self.weights = nn.Parameter(torch.empty((1, self.weightTensorDepth, self.spatialBlocksSize, self.inChannels),requires_grad=True, dtype=torch.float))
        torch.nn.init.xavier_uniform_(self.weights)
        self.bias = nn.Parameter(torch.empty((1, outChannels, self.outputHeight, self.outputWidth),requires_grad=True, dtype=torch.float))
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, input):
        # Perform Col2Vol operation on the input feature given kernel, stride, padding and dilation size
