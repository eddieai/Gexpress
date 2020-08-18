# G'express
1. Install mediapipe on Ubuntu 18.04.  Make sure to install all related packages (Bazel, OpenCV, OpenGL...)
2. Copy all files in **mediapipe/mediapipe/examples/desktop/** of this repository into the **same path** of mediapipe project folder.
3. Build Mediapipe multiple hand landmark GPU with output:
```
bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS mediapipe/examples/desktop/multi_hand_tracking:multi_hand_tracking_landmarks_gpu
```
4. Run:
```
./bazel-bin/mediapipe/examples/desktop/multi_hand_tracking//multi_hand_tracking_landmarks_gpu --calculator_graph_config_file=mediapipe/graphs/hand_tracking/multi_hand_tracking_mobile.pbtxt
```
5. Landmarks are exported in **mediapipe/output.txt** in real time
6. Use **load_mediapipe_out.py** load read output.txt into Python in real time

Update：
Follow the tutorial to install python wrapper of hand tracking model: https://github.com/metalwhale/hand_tracking
To extract the images of hand gesture, run get_hand_keypoints.py, example of dataset coming from https://www.kaggle.com/muhammadkhalid/sign-language-for-numbers
