from flask import Flask, render_template, Response
import cv2
import numpy as np
from src.hand_tracker import HandTracker
from scipy.signal import resample
from inference import *

WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "models/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"
ANCHORS_PATH = "models/anchors.csv"

label = ['Swiping Left',
         'Swiping Right',
         'Swiping Down',
         'Swiping Up',
         'Pushing Hand Away',
         'Pulling Hand In',
         'Sliding Two Fingers Left',
         'Sliding Two Fingers Right',
         'Sliding Two Fingers Down',
         'Sliding Two Fingers Up',
         'Pushing Two Fingers Away',
         'Pulling Two Fingers In',
         'Rolling Hand Forward',
         'Rolling Hand Backward',
         'Turning Hand Clockwise',
         'Turning Hand Counterclockwise',
         'Zooming In With Full Hand',
         'Zooming Out With Full Hand',
         'Zooming In With Two Fingers',
         'Zooming Out With Two Fingers',
         'Thumb Up',
         'Thumb Down',
         'Shaking Hand',
         'Stop Sign',
         'Drumming Fingers',
         'No gesture',
         'Doing other things']

app = Flask(__name__)

param_x = 0
param_y = 0
param_class = 0

class VideoCamera(object):
    def __init__(self):
        self.box_x = 0
        # self.video = cv2.VideoCapture('snowing.mp4')
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FPS, 30)
        self.frame_iter = 0
        self.max_frame_iter = 10
        self.frame_slide = 5
        self.data_window = np.zeros((0, 42))
        self.detector = HandTracker(
            PALM_MODEL_PATH,
            LANDMARK_MODEL_PATH,
            ANCHORS_PATH,
            box_shift=0.2,
            box_enlarge=1.3
        )

    def __del__(self):
        self.video.release()

    def get_frame(self):
        global param_x
        global param_y
        global param_w
        #global param_h
        global param_class
        success, frame = self.video.read()
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        height, width, channels = image.shape
        kp, box = self.detector(image)

        if kp is not None:
            keypoint = (kp / (width, height)).flatten()
            box_x = box.mean(axis=0)[0]
            box_y = box.mean(axis=0)[1]
            box_w = np.sqrt(np.sum(np.square(box[3] - box[0])))
            param_x = box_x
            param_y = box_y
            param_w = box_w
            #param_h = box_h
            #print("box_w points: \n" + str(box_w))
            #print("box_h points: \n" + str(box_h))
            self.data_window = np.vstack((self.data_window, keypoint))
            self.frame_iter = self.frame_iter + 1

        if self.frame_iter == self.max_frame_iter:
            data_window_resample = resample(self.data_window, num=100, axis=0).reshape(1, 100, 42)
            # print("Hand keypoints: \n" + str(data_window_resample))
            detected_class = inference(data_window_resample)
            param_class = detected_class
            #param_class = label[detected_class]
            #print("Detected_class: \n" + str(label[detected_class]))
            self.frame_iter = self.max_frame_iter - self.frame_slide
            self.data_window = self.data_window[self.frame_slide:]

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

@app.route('/')
def index():
    return render_template('demo.html')

def gen(camera):
    global param_y
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
   return Response(gen(VideoCamera()),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/updatex', methods=['GET'])
def updatex():
    global param_x
    return render_template('updatex.html', param_x = param_x)

@app.route('/updatey', methods=['GET'])
def updatey():
    global param_y
    return render_template('updatey.html', param_y = param_y)

@app.route('/updateclass', methods=['GET'])
def updateclass():
    global param_class
    return render_template('updateclass.html', param_class = param_class)

@app.route('/updatew', methods=['GET'])
def updatew():
    global param_w
    return render_template('updatew.html', param_w = param_w)

@app.route('/updateh', methods=['GET'])
def updateh():
    global param_h
    return render_template('updateh.html', param_h = param_h)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)

