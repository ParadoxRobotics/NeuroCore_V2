import cv2
import numpy as np
import math
from imutils.video import WebcamVideoStream

print('init camera')
cam = WebcamVideoStream(src=0).start()

integral_input = np.zeros((480,640))
tau = 0.1

while True:

    state = cam.read()
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = cv2.resize(state, (640,480))

    integrator = tau*integral_input + (1-tau)*state
    activation = state - integrator

    integral_input = integrator

    cv2.imshow('leaky integrator', activation)

    if cv2.waitKey(1) == 27:
        break
