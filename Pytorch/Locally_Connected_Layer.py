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
        outputSize = self.compute_output_size(inputSize, inputKernelSize, strideSize, paddingMode)
        # compute axes ticks
        outputAxesTicks = [range(d) for d in outputSize]
        # for an input shape [H, W, C] with channel in last -> concat index (lambda generator)
        concatIdxs = lambda spatial_idx, filter_idx: spatial_idx + (filter_idx,)

        for outputPosition in itertools.product(*outputAxesTicks):
            inputAxesTicks = self.location_output_input(inputSize, inputKernelSize, strideSize, outputPosition, paddingMode)
            for inputPosition in itertools.product(*inputAxesTicks):
                for fanIn in range(inputChannels):
                    for fanOut in range(outputChannels):
                        outIdx = np.ravel_multi_index(multi_index=concatIdxs(outputPosition, fanOut), dims=concatIdxs(outputSize, outputChannels))
                        inIdx = np.ravel_multi_index(multi_index=concatIdxs(inputPosition, fanIn), dims=concatIdxs(inputSize, inputChannels))
                        # generator containing all output-input index for a sparse matrix
                        yield (outIdx, inIdx)

    def __init__(self, inputSize, inputKernelSize, recurrentKernelSize, inputChannels, outputChannels, upperHiddenSize, upperKernelSize, upperHiddenChannels):
        super(PredictiveLayer, self).__init__()

        # create attribut from the user :
        self.inputSize = inputSize # [H, W]
        self.inputKernelSize = inputKernelSize # [H, W]
        self.recurrentKernelSize = recurrentKernelSize # [H, W]
        self.inputChannels = inputChannels # int
        self.outputChannels = outputChannels # int

        if len(upperHiddenSize) == len(upperHiddenSize) == len(upperHiddenChannels):
            # generate n feedback connection
            self.upperHiddenSize = upperHiddenSize # [H, W]
            self.upperKernelSize = upperKernelSize # [H, W]
            self.upperHiddenChannels = upperHiddenChannels # int

            self.numberFeedback = len(upperHiddenSize)

            # compute and create internal attribut for the layer :
            # calculate output size (output = [H, w]) corresponding to the hidden representation of the layer
            self.outputSize = self.compute_output_size(self.inputSize, self.inputKernelSize, self.inputKernelSize, paddingMode=False)

            # calculate weight shape for every I/O of the layer
            # encoder / recurrence / decoder
            self.WInputSize = (self.outputSize[0]*self.outputSize[1]*self.outputChannels, self.inputSize[0]*self.inputSize[1]*self.inputChannels)
            self.WErrorSize = (self.outputSize[0]*self.outputSize[1]*self.outputChannels, self.inputSize[0]*self.inputSize[1]*self.inputChannels)
            self.WRecurrentSize = (self.outputSize[0]*self.outputSize[1]*self.outputChannels, self.outputSize[0]*self.outputSize[1]*self.outputChannels)
            self.WDecoderSize = (self.outputSize[0]*self.outputSize[1]*self.outputChannels, self.inputSize[0]*self.inputSize[1]*self.inputChannels)
            # feedback
            if self.numberFeedback != 0:
                self.WFeedbackSize = []
                for i in range(len(self.upperHiddenSize)):
                    self.WFeedbackSize.append((self.upperHiddenSize[i][0]*self.upperHiddenSize[i][1]*self.upperHiddenChannels[i], self.outputSize[0]*self.outputSize[1]*self.outputChannels))

            # compute the weight kernel index (list of tuple) for filling the sparse matrix with the weight stored in a 1D array
            # encoder / recurrence / decoder
            self.WInputIdx = sorted(self.kernel_index(self.inputSize, self.inputKernelSize, self.inputKernelSize, False, self.inputChannels, self.outputChannels))
            self.WErrorIdx = sorted(self.kernel_index(self.inputSize, self.inputKernelSize, self.inputKernelSize, False, self.inputChannels, self.outputChannels))
            self.WRecurrentIdx = sorted(self.kernel_index(self.outputSize, self.recurrentKernelSize, (1,1), True, self.outputChannels, self.outputChannels))
            self.WDecoderIdx = sorted(self.kernel_index(self.inputSize, self.inputKernelSize, self.inputKernelSize, False, self.inputChannels, self.outputChannels))
            # feedback
            if self.numberFeedback != 0:
                self.WFeedbackIdx = []
                for i in range(self.numberFeedback):
                    self.WFeedbackIdx.append(sorted(self.kernel_index(self.outputSize, self.upperKernelSize[i], self.upperKernelSize[i], False, self.inputChannels, self.upperHiddenChannels[i])))

        else:
            raise Exception('not same feedback parameters size')


# test
Layer_1 = PredictiveLayer(inputSize=(96,96), inputKernelSize=(6,6), recurrentKernelSize=(4,4), inputChannels=3, outputChannels=49, upperHiddenSize=[(8,8)], upperKernelSize=[(2,2)],  upperHiddenChannels=[49])
