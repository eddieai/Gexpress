import cv2
import glob
import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from hand_tracker import HandTracker
from matplotlib.patches import Polygon



palm_model_path = "./models/palm_detection_without_custom_op.tflite"
landmark_model_path = "./models/hand_landmark.tflite"
anchors_path = "./data/anchors.csv"


# Gesture folder loacation
folderLocation = './data/sign_language/'
gesture = folderLocation[folderLocation[:-1].rfind('/')+1:-1]

classNum_start = 0

frameNum = 1
classNum = classNum_start
keypoints = []

# box_shift determines
detector = HandTracker(palm_model_path, landmark_model_path, anchors_path,
                       box_shift=0.2, box_enlarge=1.3)

while True:
    try:
        # Read frames on directory
        imgPath = sorted(glob.glob(folderLocation + str(classNum) + '/*.jpg'),
                         key=lambda x: int(re.match(r'.*?(\d{1,3})\.jpg$', x).group(1)))

        img = cv2.imread(imgPath[frameNum-1])[:,:,::-1]

        print('Frame path: \t\t', imgPath[frameNum - 1])

        # Process and display images
        X = np.empty((0))
        Y = np.empty((0))

        kp, box = detector(img)

        # f, ax = plt.subplots(1, 1, figsize=(10, 10))
        # ax.imshow(img)
        # ax.scatter(kp[:, 0], kp[:, 1])
        # ax.add_patch(Polygon(box, color="#00ff00", fill=False))

        if kp is not None:
            print('Mediapipe keypoints shape of Class %d Frame %d: ' % (classNum, frameNum), kp.shape, '\n')
            # print("Hand keypoints: \n" + str(kp))

            keypoint = kp.flatten()
            boxpoint = box.flatten()

            keypoints.append({
                'Frame No.': frameNum,
                'Keypoints': keypoint.tolist(),
                'Box': boxpoint.tolist(),
                'Class': classNum,
                })

            frameNum += 1
            if frameNum > len(imgPath):
                classNum += 1
                frameNum = 1

            if not (os.path.exists(folderLocation + str(classNum))):
                print('Sub folder %d not found' % (classNum))
                break

    except Exception as e:
        frameNum += 1
        if frameNum > len(imgPath):
            classNum += 1
            frameNum = 1

        if not (os.path.exists(folderLocation + str(classNum))):
            print('Sub folder %d not found' % (classNum))
            break
        pass
    continue

for items in keypoints:
    print(items)

with open('keypoints_dataset_sign_language_class%d-%d.json' % (classNum_start, classNum-1), 'w') as json_file:
    json.dump(keypoints, json_file)