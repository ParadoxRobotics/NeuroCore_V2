import cv2
import numpy as np
import math
from imutils.video import WebcamVideoStream

print('init camera')
cam = WebcamVideoStream(src=0).start()

state = cam.read()
state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
state = cv2.resize(state, (128,128))

integral_input = state
integral_input_ = state
tau = 0.01

while True:

    state = cam.read()
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = cv2.resize(state, (128,128))

    integrator = tau*state + (1-tau)*integral_input
    integrator_ = tau*integral_input_ + (1-tau)*state
    
    int_input = state - integrator
    int_input_ = state - integrator_

    integral_input = integrator.copy()
    integral_input_ = integrator_.copy()

    cv2.imshow('leaky integrator', cv2.applyColorMap(np.uint8(cv2.normalize(int_input, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)), cv2.COLORMAP_JET))
    #cv2.imshow('leaky integrator', cv2.applyColorMap(np.uint8(cv2.normalize(int_input_, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)), cv2.COLORMAP_JET))

    if cv2.waitKey(1) == 27:
        break
