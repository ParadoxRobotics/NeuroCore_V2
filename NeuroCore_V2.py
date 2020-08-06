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
import matplotlib.pyplot as plt
import math
import cv2
# Pytorch library
import torch
import torch.nn as nn
from torch.nn import init
# custom layer
from Locally_Connected_Layer import *
