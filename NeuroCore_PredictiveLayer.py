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
import copy
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
            outputSize = [(((inputSize[idx]-inputKernelSize[idx]+1)+strideSize[idx]-1)//strideSize[idx]) for idx in range(len(inputSize))]
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

        # check size input
        if kernelDim == inDim and strideDim == inDim:
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
        else:
            raise Exception('kernel dimension or stride dimension must agree with input dimension')

    # weight initilization with lecun_normal init
    def weight_init_normal(self, weightSize, weightIndexSize):
        # mean = 0 and stddev = sqrt(1/inputSize)
        np.random.seed(42)
        return np.random.randn(weightIndexSize,)*np.sqrt(1/weightSize)

    # bias initialization at 1
    def bias_init(self, outputSize, outputChannels):
        return np.ones((outputSize[0], outputSize[1], outputChannels))

    # ReLU activation function
    def ReLU(self, input):
        return input*(input>0)

    # ReLU derivative for backpropagation
    def dt_ReLU(self, input):
        return (input>0)*1

    # SELU activation function
    def SELU(self, input):
        alphaValue = 1.6733
        lambdaValue = 1.0507
        if input <= 0.0:
            return lambdaValue*(alphaValue*np.exp(input)-alphaValue)
        else:
            return lambdaValue*input

    # SELU derivative for backpropagation
    def dt_SELU(self, input):
        alphaValue = 1.6733
        lambdaValue = 1.0507
        if input <= 0.0:
            return lambdaValue*alphaValue*np.exp(input)
        else:
            return lambdaValue

    # Leaky integrator neuron with ReLU activation
    def LIR(self, input, prevAct, tau):
        act = tau*input+(1-tau)*prevAct
        return max(0, act), act

    # Leaky integrator neuron derivative
    def dt_LIR(self, input):
        return (input>0)*1

    # leaky integrator neuron with global K% winner-take-all
    def LIR_WTA(self, input, prevAct, tau, sparsity):
        # Leaky integrator neuron
        act = tau*input+(1-tau)*prevAct
        memAct = act.copy()
        # compute full WTA activation
        kp = int(len(act)*sparsity)
        threshold = sorted(act)[kp]
        actSparse = np.array([float(x>=threshold) for x in act])
        actSparse = np.reshape(actSparse, (len(actSparse),1))
        return np.where(actSparse == 1, act, 0), memAct

    # memory trace
    def mem_trace(self, input, prevState, tau):
        return tau*prevState+(1-tau)*input


    def __init__(self, inputSize, inputKernelSize, recurrentKernelSize, inputChannels, outputChannels, upperHiddenSize, upperKernelSize, upperHiddenChannels, biasMode, neuronType, sparsity, tau):
        super(PredictiveLayer, self).__init__()

        # create attribut from the user :
        self.inputSize = inputSize # [H, W]
        self.inputKernelSize = inputKernelSize # [H, W]
        self.recurrentKernelSize = recurrentKernelSize # [H, W]
        self.inputChannels = inputChannels # int
        self.outputChannels = outputChannels # int
        self.biasMode = biasMode
        self.neuronType = neuronType
        self.sparsity = sparsity
        self.tau = tau

        # check if feedback parameters size are equal and non-empty
        if len(upperHiddenSize) == len(upperHiddenSize) == len(upperHiddenChannels):
            # get feedback parameters
            self.upperHiddenSize = upperHiddenSize # [H, W]
            self.upperKernelSize = upperKernelSize # [H, W]
            self.upperHiddenChannels = upperHiddenChannels # int
            self.numberFeedback = len(upperHiddenSize) # int

            # compute and create internal attribut for the layer :
            # calculate output size (output = [H, w]) corresponding to the hidden representation of the layer
            self.outputSize = self.compute_output_size(self.inputSize, self.inputKernelSize, self.inputKernelSize, paddingMode=False)
            print(self.outputSize)
            # calculate weight shape for every I/O of the layer
            # encoder / recurrence / decoder
            self.WInputSize = (self.outputSize[0]*self.outputSize[1]*self.outputChannels, self.inputSize[0]*self.inputSize[1]*self.inputChannels)
            self.WErrorSize = (self.outputSize[0]*self.outputSize[1]*self.outputChannels, self.inputSize[0]*self.inputSize[1]*self.inputChannels)
            # replace the recurrent lateral layer by a winner-take-all Leaky integrator neuron
            if neuronType != 'LIR_WTA' and sparsity == None:
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
            if neuronType != 'LIR_WTA' and sparsity == None:
                self.WRecurrentIdx = sorted(self.kernel_index(self.outputSize, self.recurrentKernelSize, (1,1), True, self.outputChannels, self.outputChannels))
            self.WDecoderIdx = sorted(self.kernel_index(self.inputSize, self.inputKernelSize, self.inputKernelSize, False, self.inputChannels, self.outputChannels))
            # feedback
            if self.numberFeedback != 0:
                self.WFeedbackIdx = []
                for i in range(self.numberFeedback):
                    self.WFeedbackIdx.append(sorted(self.kernel_index(self.outputSize, self.upperKernelSize[i], self.upperKernelSize[i], False, self.inputChannels, self.upperHiddenChannels[i])))

            # Create weight vector for the sparse weight matrix
            # encoder / recurrence / decoder
            self.WInput = self.weight_init_normal(weightSize=self.WInputSize[1], weightIndexSize=len(self.WInputIdx))
            self.WError = self.weight_init_normal(weightSize=self.WErrorSize[1], weightIndexSize=len(self.WErrorIdx))
            if neuronType != 'LIR_WTA' and sparsity == None:
                self.WRecurrent = self.weight_init_normal(weightSize=self.WRecurrentSize[1], weightIndexSize=len(self.WRecurrentIdx))
            self.WDecoder = self.weight_init_normal(weightSize=self.WDecoderSize[0], weightIndexSize=len(self.WDecoderIdx))
            # feedback
            if self.numberFeedback != 0:
                self.WFeedback = []
                for i in range(self.numberFeedback):
                    self.WFeedback.append(self.weight_init_normal(weightSize=self.WFeedbackSize[i][0], weightIndexSize=len(self.WFeedbackIdx[i])))

            # create bias for the hidden layer
            if self.biasMode == True:
                self.bias = self.bias_init(self.outputSize, self.outputChannels)

        else:
            raise Exception('not same feedback parameters size')

# test
Layer_1 = PredictiveLayer(inputSize=(96,96),
                          inputKernelSize=(6,6),
                          recurrentKernelSize=(4,4),
                          inputChannels=3, outputChannels=49,
                          upperHiddenSize=[(8,8), (4,4), (2,2), (1,1)],
                          upperKernelSize=[(2,2), (2,2), (2,2), (1,1)],
                          upperHiddenChannels=[49, 49, 49, 49],
                          biasMode=True,
                          neuronType=None,
                          sparsity=None,
                          tau=None)

Layer_2 = PredictiveLayer(inputSize=(16,16),
                          inputKernelSize=(2,2),
                          recurrentKernelSize=(4,4),
                          inputChannels=49, outputChannels=49,
                          upperHiddenSize=[(4,4), (2,2), (1,1)],
                          upperKernelSize=[(2,2), (2,2), (1,1)],
                          upperHiddenChannels=[49, 49, 49],
                          biasMode=True,
                          neuronType=None,
                          sparsity=None,
                          tau=None)

Layer_3 = PredictiveLayer(inputSize=(8,8),
                          inputKernelSize=(2,2),
                          recurrentKernelSize=(4,4),
                          inputChannels=49, outputChannels=49,
                          upperHiddenSize=[(2,2), (1,1)],
                          upperKernelSize=[(2,2), (1,1)],
                          upperHiddenChannels=[49, 49],
                          biasMode=True,
                          neuronType=None,
                          sparsity=None,
                          tau=None)

Layer_4 = PredictiveLayer(inputSize=(4,4),
                          inputKernelSize=(2,2),
                          recurrentKernelSize=(2,2),
                          inputChannels=49, outputChannels=49,
                          upperHiddenSize=[(1,1)],
                          upperKernelSize=[(1,1)],
                          upperHiddenChannels=[49],
                          biasMode=True,
                          neuronType=None,
                          sparsity=None,
                          tau=None)

Layer_5 = PredictiveLayer(inputSize=(2,2),
                          inputKernelSize=(2,2),
                          recurrentKernelSize=(1,1),
                          inputChannels=49, outputChannels=49,
                          upperHiddenSize=[],
                          upperKernelSize=[],
                          upperHiddenChannels=[],
                          biasMode=True,
                          neuronType=None,
                          sparsity=None,
                          tau=None)
