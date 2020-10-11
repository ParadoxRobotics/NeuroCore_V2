#     _   __                      ______
#    / | / /__  __  ___________  / ____/___  ________
#   /  |/ / _ \/ / / / ___/ __ \/ /   / __ \/ ___/ _ \
#  / /|  /  __/ /_/ / /  / /_/ / /___/ /_/ / /  /  __/
# /_/ |_/\___/\__,_/_/   \____/\____/\____/_/   \___/

# NeuroCore is an PyTorch implementation of a Predictive Visual Model for robotics
# applications (tracking, recognition, manipulation,...)

# Author : Munch Quentin, 2020

import numpy as np
import math
import cv2
import itertools
from matplotlib import pyplot as plt

class PredictiveLayer():
    # internal class function for computing output size given kernel, stride, padding and input dimension
    # if paddingMode = True, the output shape will be the same as the input
    # else there is no padding (no overlapping in the receptive field)
    def compute_output_size(self, inputSize, inputKernelSize, strideSize, paddingMode):
        if paddingMode == True:
            outputSize = [((inputSize[idx]+strideSize[idx]-1)//strideSize[idx]) for idx in range(len(inputSize))]
            outputSize = tuple([0 if inputSize[idx] == 0 else outputSize[idx] for idx in range(len(inputSize))])
        else:
            outputSize = [((inputSize[idx]-inputKernelSize[idx]+1+strideSize[idx]-1)//strideSize[idx]) for idx in range(len(inputSize))]
            outputSize = tuple([0 if inputSize[idx] == 0 else outputSize[idx] for idx in range(len(inputSize))])
        return outputSize

    # internal class function for extracting the locations of the input connected to a a given output position
    def location_output_input(self, inputSize, inputKernelSize, strideSize, outputPosition, paddingMode):
        # init location
        location = []
        # input dimension
        inDim = len(inputSize)
        # for every dimension in the current input (ex HxW -> d=2)
        for d in range(inDim):
            # pos in d
            leftShift = int(inputKernelSize[d]/2)
            rightShift = inputKernelSize[d]-leftShift
            center = outputPosition[d]*strideSize[d]
            # if no padding
            if paddingMode == False:
                center += leftShift
            # get pos
            startPos = max(0, center-leftShift)
            endPos = min(inputSize[d], center+rightShift)
            # append location
            location.append(range(startPos, endPos))
        # return position
        return location

    # internal class function for computing the weight index in the sparse matrix representing the locally connected operation
    def kernel_index(self, inputSize, inputKernelSize, strideSize, paddingMode, inputChannels, outputChannels):
        # input dimension
        inDim = len(inputSize)
        # kernel and stride dimension
        kernelDim = len(inputKernelSize)
        strideDim = len(strideSize)
        # compute output shape
        outputSize = compute_output_shape(inputSize, inputKernelSize, strideSize, paddingMode)
        # compute axes ticks
        outputAxesTicks = [range(d) for d in outputSize]
        # for an input shape [H, W, C] with channel in last -> concat index (lambda generator)
        concatIdxs = lambda spatial_idx, filter_idx: spatial_idx + (filter_idx,)

        for outputPosition in itertools.product(*outputAxesTicks):
            inputAxesTicks = location_output_input(inputSize, inputKernelSize, strideSize, outputPosition, paddingMode)
            for inputPosition in itertools.product(*inputAxesTicks):
                for fanIn in range(inputChannels):
                    for fanOut in range(outputChannels):
                        outIdx = np.ravel_multi_index(multi_index=concatIdxs(outputPosition, fanOut), dims=concatIdxs(outputSize, outputChannels))
                        inIdx = np.ravel_multi_index(multi_index=concatIdxs(inputPosition, fanIn), dims=concatIdxs(inputSize, inputChannels))
                        # generator containing all output-input index for a sparse matrix
                        yield (outIdx, inIdx)

    # weight initialization using

    def __init__(self, inputSize, inputKernelSize, recurrentKernelSize, inputChannels, outputChannels, upperHiddenSize, upperKernelSize, upperHiddenChannels):
        super(PredictiveLayer, self).__init__()

        # create attribut from the user :
        self.inputSize = inputSize # [H, W]
        self.inputKernelSize = inputKernelSize # [H, W]
        self.recurrentKernelSize = recurrentKernelSize # [H, W]
        self.inputChannels = inputChannels # int
        self.outputChannels = outputChannels # int
        self.upperHiddenSize = upperHiddenSize # [H, W]
        self.upperKernelSize = upperKernelSize # [H, W]
        self.upperHiddenChannels = upperHiddenChannels # int

        # compute and create internal attribut for the layer :
        # calculate output size (output = [H, w]) corresponding to the hidden representation of the layer
        self.outputSize = self.compute_output_size(self.inputSize, self.inputKernelSize, self.inputKernelSize, paddingMode=False)

        # calculate weight shape for every I/O of the layer
        # main input encoder (Hin x Win x Cin, Hout x Wout x Cout)
        self.WISize = (self.outputSize[0]*self.outputSize[1]*self.outputChannels, self.inputSize[0]*self.inputSize[1]*self.inputChannels)
        # error input encoder (Hin x Win x Cin, Hout x Wout x Cout)
        self.WESize = (self.outputSize[0]*self.outputSize[1]*self.outputChannels, self.inputSize[0]*self.inputSize[1]*self.inputChannels)
        # internal recurrent connection (Hout x Wout x Cout, Hout x Wout x Cout)
        self.WRSize = (self.outputSize[0]*self.outputSize[1]*self.outputChannels, self.outputSize[0]*self.outputSize[1]*self.outputChannels)
        # feedback connection encoder (Hin x Win x Cin, Hout x Wout x Cout) -> (Hout x Wout x Cout, Hin x Win x Cin)
        # (since the sparse matrix is reversible we can use the same operation for feedback projection)
        self.WFSize = (self.upperHiddenSize[0]*self.upperHiddenSize[1]*self.upperHiddenChannels, self.outputSize[0]*self.outputSize[1]*self.outputChannels)
        # main output decoder (Hin x Win x Cin, Hout x Wout x Cout) -> (Hout x Wout x Cout, Hin x Win x Cin)
        # (since the sparse matrix is reversible we can use the same operation for decoding)
        self.WDSize = (self.outputSize[0]*self.outputSize[1]*self.outputChannels, self.inputSize[0]*self.inputSize[1]*self.inputChannels)

        # compute the weight kernel index (list of tuple) for filling the sparse matrix with the weight stored in a 1D array
        # main input
        self.WIIdx = self.kernel_index(self.inputSize, self.inputKernelSize, self.inputKernelSize, False, self.inputChannels, self.outputChannels)
        # error input
        self.WEIdx = self.kernel_index(self.inputSize, self.inputKernelSize, self.inputKernelSize, False, self.inputChannels, self.outputChannels)
        # recurrent connection
        self.WRIdx = self.kernel_index(self.outputSize, self.recurrentKernelSize, (1,1), True, self.outputChannels, self.outputChannels)
        # feedback input
        self.WFIdx = self.kernel_index(self.outputSize, self.upperKernelSize, self.upperKernelSize, False, self.inputChannels, self.upperHiddenChannels)
        # decoder output
        self.WDIdx = self.kernel_index(self.inputSize, self.inputKernelSize, self.inputKernelSize, False, self.inputChannels, self.outputChannels)
        """
        # create weight matrix
        self.WI =
        self.WE =
        self.WR =
        self.WF =
        self.WD =
        """

# test
Layer_1 = PredictiveLayer(inputSize=(96,96), inputKernelSize=(6,6), recurrentKernelSize=(4,4), inputChannels=3, outputChannels=49, upperHiddenSize=(8,8), upperKernelSize=(2,2),  upperHiddenChannels=49)
