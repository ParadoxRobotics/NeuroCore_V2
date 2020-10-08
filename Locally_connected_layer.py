import numpy as np
import math
import itertools

def compute_output_shape(inputShape, kernelShape, strideShape, paddingMode):
    if paddingMode == True:
        return  [((inputShape[index]+strideShape[index]-1)//strideShape[index]) for index in range(len(inputShape))]
    else:
        return [((inputShape[index]-kernelShape[index]+1+strideShape[index]-1)//strideShape[index]) for index in range(len(inputShape))]


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
    # get kernel/stride size over the input dimension
    kernelShape = int((kernelShape,))*inDim
    strideShape = int((strideShape,))*inDim
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
                    inIdx = np.ravel_multi_index(multi_index=concat_idxs(inputPosition, fanIn), dims=concat_idxs(inputShape, channelIn))
                    yield (outIdx, inIdx)
