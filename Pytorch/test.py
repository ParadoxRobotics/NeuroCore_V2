import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
# custom layer
from Predictive_Layer import *

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

a = list(LPU_1.parameters())[0].clone()

optim_1 = optim.SGD(LPU_1.parameters(), lr=0.01, momentum=0.90, weight_decay=0.00001, nesterov=True)

cost = nn.MSELoss(reduction = 'sum')

lastHt_1 = torch.randn(1,49,64,64,requires_grad=True)
lastPred_1 = torch.randn(1,3,128,128,requires_grad=True)
cmd = torch.zeros(1,49,32,32,requires_grad=True)

Xt = torch.randn(1,3,128,128,requires_grad=True)
Et_1 = torch.randn(1,3,128,128,requires_grad=True)

yt, ht = LPU_1(Xt, Et_1, lastHt_1, cmd)

lastPred_1 = yt
lastHt_1 = ht

LPU_1.train()
for i in range(0,5):
    Xt = torch.randn(1,3,128,128,requires_grad=True)
    Et_1 = torch.randn(1,3,128,128,requires_grad=True)

    yt, ht = LPU_1(Xt, Et_1, lastHt_1, cmd)

    optim_1.zero_grad()
    cost_1 = cost(lastPred_1, Xt)
    print(cost_1)
    cost_1.backward(retain_graph=True)
    torch.autograd.set_detect_anomaly(True)
    optim_1.step()

    b = list(LPU_1.parameters())[0].clone()

    print(torch.equal(a,b))

    lastPred_1 = yt
    lastHt_1 = ht

