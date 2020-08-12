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








import torch.optim as optim
layer = NeuroCore_A()

a = list(layer.parameters())[0].clone()

Xt = torch.randn(1,3,128,128,requires_grad=True)
# error of prediction
Et_1 = torch.randn(1,3,128,128,requires_grad=True)
Et_2 = torch.randn(1,49,64,64,requires_grad=True)
Et_3 = torch.randn(1,49,32,32,requires_grad=True)
Et_4 = torch.randn(1,49,16,16,requires_grad=True)
Et_5 = torch.randn(1,49,8,8,requires_grad=True)
# last state
lastHt_1 = torch.randn(1,49,64,64,requires_grad=True)
lastHt_2 = torch.randn(1,49,32,32,requires_grad=True)
lastHt_3 = torch.randn(1,49,16,16,requires_grad=True)
lastHt_4 = torch.randn(1,49,8,8,requires_grad=True)
lastHt_5 = torch.randn(1,49,4,4,requires_grad=True)
# upper command fedback
cmd = torch.zeros(1,1,1,1,requires_grad=True)

lastPred_1 = torch.randn(1,3,128,128,requires_grad=True)
lastPred_2 = torch.randn(1,49,64,64,requires_grad=True)
lastPred_3 = torch.randn(1,49,32,32,requires_grad=True)
lastPred_4 = torch.randn(1,49,16,16,requires_grad=True)
lastPred_5 = torch.randn(1,49,8,8,requires_grad=True)




optim_1 = optim.SGD(list(layer.parameters()), lr=1, momentum=0.90, weight_decay=0.00001, nesterov=True)
cost = nn.MSELoss()


layer.train()
Yt_1, Yt_2, Yt_3, Yt_4, Yt_5, Ht_1, Ht_2, Ht_3, Ht_4, Ht_5 = layer(Xt, Et_1, Et_2, Et_3, Et_4, Et_5, lastHt_1, lastHt_2, lastHt_3, lastHt_4, lastHt_5, cmd)
print(Yt_1.shape)

optim_1.zero_grad()
cost_1 = cost(Yt_1, Xt)
cost_1.register_hook(lambda grad: print(grad))
cost_1.backward()


optim_1.step()

b = list(layer.parameters())[0].clone()

print(torch.equal(a,b))
print(a-b)

