import numpy as np
import math
import itertools

def compute_output_shape(inputShape, kernelShape, strideShape, paddingMode):
    if paddingMode == True:
        outShape = [((inputShape[index]+strideShape[index]-1)//strideShape[index]) for index in range(len(inputShape))]
        outShape = tuple([0 if inputShape[index] == 0 else outShape[index] for index in range(len(inputShape))])
    else:
        outShape = [((inputShape[index]-kernelShape[index]+1+strideShape[index]-1)//strideShape[index]) for index in range(len(inputShape))]
        outShape = tuple([0 if inputShape[index] == 0 else outShape[index] for index in range(len(inputShape))])
    return outShape

# Return locations of the input connected to an output position.
def location_output_input(inputShape, kernelShape, strideShape, outPosition, paddingMode):
    # init location
    location = []
    # input dimension
    inDim = len(inputShape)
    # for every dimension in the current input (ex HxW -> d=2)
    for d in range(inDim):
        # pos in d
        leftShift = int(kernelShape[d]/2)
        rightShift = kernelShape[d]-leftShift
        center = outPosition[d]*strideShape[d]
        # if no padding
        if paddingMode == False:
            center += leftShift
        # get pos
        startPos = max(0, center-leftShift)
        endPos = min(inputShape[d], center+rightShift)
        # append location
        location.append(range(startPos, endPos))
    # return position
    return location

# Yields output-input tuples of indices in a layer.
def kernel_index(inputShape, kernelShape, strideShape, paddingMode, channelIn, channelOut):
    # input dimension
    inDim = len(inputShape)
    # kernel and stride dimension
    kernelDim = len(kernelShape)
    strideDim = len(strideShape)
    # compute output shape
    outputShape = compute_output_shape(inputShape, kernelShape, strideShape, paddingMode)
    # compute axes ticks
    outputAxesTicks = [range(d) for d in outputShape]
    # for an input shape [H, W, C] with channel in last -> concat index (lambda generator)
    concatIdxs = lambda spatial_idx, filter_idx: spatial_idx + (filter_idx,)

    for outputPosition in itertools.product(*outputAxesTicks):
        inputAxesTicks = location_output_input(inputShape, kernelShape, strideShape, outputPosition, paddingMode)
        for inputPosition in itertools.product(*inputAxesTicks):
            for fanIn in range(channelIn):
                for fanOut in range(channelOut):
                    outIdx = np.ravel_multi_index(multi_index=concatIdxs(outputPosition, fanOut), dims=concatIdxs(outputShape, channelOut))
                    inIdx = np.ravel_multi_index(multi_index=concatIdxs(inputPosition, fanIn), dims=concatIdxs(inputShape, channelIn))
                    yield (outIdx, inIdx)


# perform local computation with input shape = [1, H, W, C]
def compute_LC(input, kernelWeight, kernelIdx, KernelShape, outputShape):
    inputFlat = np.reshape(input, (input.shape[0], -1))
    return None



inputSize = (96,96)
kernelSize = (6,6)
strideSize = (6,6)
filterIn = 3
filterOut = 49
padding = False

outputShape = compute_output_shape(inputShape=inputSize, kernelShape=kernelSize, strideShape=strideSize, paddingMode=padding)
kernelShape = (outputShape[0] * outputShape[1] * filterOut, inputSize[0] * inputSize[1] * filterIn)
KernelIdx = sorted(kernel_index(inputShape=inputSize, kernelShape=kernelSize, strideShape=strideSize, paddingMode=padding, channelIn=filterIn, channelOut=filterOut))
kernelWeight = np.random.randn(len(KernelIdx),)
bias = np.random.randn(outputShape[0], outputShape[1], filterOut)

print(inputSize, kernelSize, strideSize, filterIn, filterOut, padding)
print(outputShape)
print(kernelShape)
print(kernelWeight.shape)
print(bias.shape)

inputFlat = np.random.randn(1,96*96*3)
sparseMat = np.zeros(kernelShape)

for w in range(0,len(KernelIdx)):
    value = kernelWeight[w,]
    idx = KernelIdx[w]
    sparseMat[idx[0], idx[1]] = value

outputFlat = np.dot(inputFlat, sparseMat.T)
print(outputFlat.shape)
