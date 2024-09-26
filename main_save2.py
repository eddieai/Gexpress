from flask import Flask, render_template, Response
import cv2
import numpy as np
from src.hand_tracker import HandTracker

WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "models/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"
ANCHORS_PATH = "models/anchors.csv"

app = Flask(__name__)

param_x = 0
param_y = 36
param_class = 0

class VideoCamera(object):
    def __init__(self):
        self.box_x = 0
        # self.video = cv2.VideoCapture('snowing.mp4')
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FPS, 30)
        self.frame_iter = 0
        self.max_frame_iter = 30
        self.frame_slide = 10
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

    # def get_frame(self):
    #     success, frame = self.video.read()
    #     ret, jpeg = cv2.imencode('.jpg', frame)
    #     return jpeg.tobytes()

    def get_frame(self):
        global param_x
        global param_y
        global param_class
        success, frame = self.video.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = image.shape
        kp, box = self.detector(image)

        if kp is not None:
            keypoint = (kp / (width, height)).flatten()
            box_x = box.mean(axis=0)[0]
            box_y = box.mean(axis=0)[1]
            param_x = box_x
            param_y = box_y
            print("box_x points: \n" + str(box_x))
            print("box_y points: \n" + str(box_y))
            print(param_y)
            self.data_window = np.vstack((self.data_window, keypoint))
            self.frame_iter = self.frame_iter + 1

        if self.frame_iter == self.max_frame_iter:
            print("Hand keypoints: \n" + str(self.data_window))
            detected_class = 1
            param_class = detected_class
            print("Detected_class: \n" + str(detected_class))
            self.frame_iter = self.max_frame_iter - self.frame_slide

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

@app.route('/')
def index():
    global param_x
    global param_y
    global param_class
    return render_template('index.html', param_x = param_x, param_y = param_y, param_class = param_class)

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

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)

