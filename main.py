# Import libraries
import cv2
import mediapipe as mp

draw = mp.solutions.drawing_utils
detector = mp.solutions.face_detection

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4, 480)

with detector.FaceDetection(min_detection_confidence=0.5) as detector:
    while True:
        success, img_rgb = cap.read()
        results = detector.process(img_rgb)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * img_rgb.shape[1])
                w = int(bbox.width * img_rgb.shape[1])
                y = int(bbox.ymin * img_rgb.shape[0])
                h = int(bbox.height * img_rgb.shape[0])

            imgCrop = img_rgb[y:y+h, x:x+w]
            imgBlur = cv2.blur(imgCrop,(35,35))
            img_rgb[y:y + h, x:x + w] = imgBlur
            cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (34,59,244), 2)

        cv2.imshow("Face Blurring", img_rgb)

        # Press 'x' to exit
        if cv2.waitKey(1) == ord('x'):
            break

        cv2.waitKey(1)




