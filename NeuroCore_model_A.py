#     _   __                      ______
#    / | / /__  __  ___________  / ____/___  ________
#   /  |/ / _ \/ / / / ___/ __ \/ /   / __ \/ ___/ _ \
#  / /|  /  __/ /_/ / /  / /_/ / /___/ /_/ / /  /  __/
# /_/ |_/\___/\__,_/_/   \____/\____/\____/_/   \___/

# NeuroCore is an PyTorch implementation of a Predictive Visual Network for robotics
# applications (tracking, recognition, manipulation,...)

# Author : Munch Quentin, 2020

# general library
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import pyplot
# Pytorch library
import torch
import torch.nn as nn
from torch.nn import init
# custom layer
from Predictive_Layer import *
