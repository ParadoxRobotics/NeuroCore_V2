import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
# custom layer
from Predictive_Layer import *

LPU_1 = PredictiveLayer(inputSize=[128,128],
                        hiddenSize=[64,64],
                        feedBackSize=[32,32],
                        inChannels=3,
                        hiddenChannels=49,
                        feedBackChannels=49,
                        kernelSize=[5,5],
                        lateralKerneSize=[5,5],
                        padding=[2,2],
                        stride=[2,2])

print(list(LPU_1.parameters())[0])

optim_1 = optim.SGD(list(LPU_1.parameters()), lr=0.1, momentum=0.90, weight_decay=0.00001, nesterov=True)

cost = nn.MSELoss(reduction = 'sum')

Xt = Variable(torch.randn(1,3,128,128),requires_grad=True)
Et_1 = Variable(torch.randn(1,3,128,128),requires_grad=True)
lastHt_1 = Variable(torch.randn(1,49,64,64),requires_grad=True)
cmd = Variable(torch.randn(1,49,32,32),requires_grad=True)
lastPred_1 = Variable(torch.randn(1,3,128,128),requires_grad=True)

yt, ht = LPU_1(Xt, Et_1, lastHt_1, cmd)

optim_1.zero_grad()
cost_1 = cost(lastPred_1, Xt)
print(cost_1)
cost_1.backward(retain_graph=True)
optim_1.step()

print(list(LPU_1.parameters())[0])
