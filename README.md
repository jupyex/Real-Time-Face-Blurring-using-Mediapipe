# Real Time Face Blurring using OpenCV and Mediapipe

Originally, I followed the [face detection guide for Python provided by Mediapipe](https://docs.readme.com/main/docs/linking-to-pages), where I created a face detector instance. Face detection was a success, yet I could not find any ways to obtain the variables `origin_x`, `origin_y`, `height` and `width` from the `FaceDetectorResult` object so as to blur the face. Thus, I resorted to using `mp.solutions.face_detection` with easy-to-use APIs to process frames and obtian face detection results, such as the coordinates of the bounding boxes around the detected faces. 

## How to use it?
Simply run the program (after installing the libraries - OpenCV and Mediapipe) and if you're in front of the camera, your face will be blurred. To stop the program from running, press `x` on your keyboard.

## Acknowledgements
The idea was inspired by [Murtaza Hassan's video](https://www.youtube.com/watch?v=cxs6iXeyfEY), yet the code is different. 
