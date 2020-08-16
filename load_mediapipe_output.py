import numpy as np
from time import time
import warnings

keypoints_old = np.zeros(0)
i = 1
while True:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            keypoints = np.loadtxt('/home/eddie/mediapipe/output.txt')
    except ValueError:
        continue

    if np.array_equal(keypoints, keypoints_old) or keypoints.size == 0:
        continue
    keypoints_old = keypoints

    print('frame %d:' % i)
    print(keypoints.shape)
    i += 1
