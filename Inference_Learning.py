#     _   __                      ______
#    / | / /__  __  ___________  / ____/___  ________
#   /  |/ / _ \/ / / / ___/ __ \/ /   / __ \/ ___/ _ \
#  / /|  /  __/ /_/ / /  / /_/ / /___/ /_/ / /  /  __/
# /_/ |_/\___/\__,_/_/   \____/\____/\____/_/   \___/

# NeuroCore is an PyTorch implementation of a Predictive Visual Network for robotics
# applications (tracking, recognition, manipulation,...)

# Author : Munch Quentin, 2020

# General library
import numpy as np
import cv2
import imutils
from imutils.video import WebcamVideoStream
import time
from matplotlib import pyplot as plt
from matplotlib import pyplot

# Pytorch library
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torch.utils.data as data
from torchvision import datasets, transforms, utils
# custom layer
from NeuroCore_Model import *

# instanciate NeuroCore model A
NCNet = NeuroCore_A()
print("Network loaded and ready")

# Create optimizer
optim_1 = optim.SGD(list(NCNet.LPU_1.parameters()), lr=0.1, momentum=0.90, weight_decay=0.00001, nesterov=True)
optim_2 = optim.SGD(list(NCNet.LPU_2.parameters()), lr=0.1, momentum=0.90, weight_decay=0.00001, nesterov=True)
optim_3 = optim.SGD(list(NCNet.LPU_3.parameters()), lr=0.1, momentum=0.90, weight_decay=0.00001, nesterov=True)
optim_4 = optim.SGD(list(NCNet.LPU_4.parameters()), lr=0.1, momentum=0.90, weight_decay=0.00001, nesterov=True)
optim_5 = optim.SGD(list(NCNet.LPU_5.parameters()), lr=0.1, momentum=0.90, weight_decay=0.00001, nesterov=True)
print("Optimizer created")

# Define cost function used
cost = nn.MSELoss(reduction = 'sum')

# test model inference + learning on random data
# input
Xt = Variable(torch.randn(1,3,128,128),requires_grad=True)
# error of prediction
Et_1 = Variable(torch.randn(1,3,128,128),requires_grad=True)
Et_2 = Variable(torch.randn(1,49,64,64),requires_grad=True)
Et_3 = Variable(torch.randn(1,49,32,32),requires_grad=True)
Et_4 = Variable(torch.randn(1,49,16,16),requires_grad=True)
Et_5 = Variable(torch.randn(1,49,8,8),requires_grad=True)
# last state
lastHt_1 = Variable(torch.randn(1,49,64,64),requires_grad=True)
lastHt_2 = Variable(torch.randn(1,49,32,32),requires_grad=True)
lastHt_3 = Variable(torch.randn(1,49,16,16),requires_grad=True)
lastHt_4 = Variable(torch.randn(1,49,8,8),requires_grad=True)
lastHt_5 = Variable(torch.randn(1,49,4,4),requires_grad=True)
# upper command fedback
cmd = Variable(torch.zeros(1,1,1,1),requires_grad=True)

lastPred_1 = Variable(torch.randn(1,3,128,128),requires_grad=True)
lastPred_2 = Variable(torch.randn(1,49,64,64),requires_grad=True)
lastPred_3 = Variable(torch.randn(1,49,32,32),requires_grad=True)
lastPred_4 = Variable(torch.randn(1,49,16,16),requires_grad=True)
lastPred_5 = Variable(torch.randn(1,49,8,8),requires_grad=True)

# set to training mode (non fixed weight)
#NCNet.train()
# perform inference
print("inference")
Yt_1, Yt_2, Yt_3, Yt_4, Yt_5, Ht_1, Ht_2, Ht_3, Ht_4, Ht_5 = NCNet(Xt, Et_1, Et_2, Et_3, Et_4, Et_5, lastHt_1, lastHt_2, lastHt_3, lastHt_4, lastHt_5, cmd)
# init optimizer + calculate loss + backprop loss + optimizer step
optim_1.zero_grad()
cost_1 = cost(lastPred_1, Xt)
print(cost_1)
cost_1.backward(retain_graph=True)
optim_1.step()
print("opt 1")
optim_2.zero_grad()
cost_2 = cost(lastPred_2, Ht_1)
print(cost_2)
cost_2.backward(retain_graph=True)
optim_2.step()
print("opt 2")
optim_3.zero_grad()
cost_3 = cost(lastPred_3, Ht_2)
print(cost_3)
cost_3.backward(retain_graph=True)
optim_3.step()
print("opt 3")
optim_4.zero_grad()
cost_4 = cost(lastPred_4, Ht_3)
print(cost_4)
cost_4.backward(retain_graph=True)
optim_4.step()
print("opt 4")
optim_5.zero_grad()
cost_5 = cost(lastPred_5, Ht_4)
print(cost_5)
cost_5.backward(retain_graph=True)
optim_5.step()
print("opt 5")

# update memory for next step
lastHt_1 = Ht_1
lastHt_2 = Ht_2
lastHt_3 = Ht_3
lastHt_4 = Ht_4
lastHt_5 = Ht_5

# compute error for the next step
Et_1 = (lastPred_1-Xt)**2
Et_2 = (lastPred_2-Ht_1)**2
Et_3 = (lastPred_3-Ht_2)**2
Et_4 = (lastPred_4-Ht_3)**2
Et_5 = (lastPred_5-Ht_4)**2

# update prediction
lastPred_1 = Yt_1
lastPred_2 = Yt_2
lastPred_3 = Yt_3
lastPred_4 = Yt_4
lastPred_5 = Yt_5

print("test over")
