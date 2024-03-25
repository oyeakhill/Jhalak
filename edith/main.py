import numpy as np
import time
import cv2
import os
import imutils
from imutils.video import VideoStream
import subprocess
from gtts import gTTS
from picamera import PiCamera
import multithread

# load the COCO class labels our YOLO model was trained on
LABELS = open("coco.names").read().strip().split("\n")

# load our YOLO object detector trained on COCO dataset (80 classes)
# For tiny-weights, use model2-files along with weights given in README.md as Model2 link and put their names in the line 18.
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
# net = cv2.dnn.readNetFromDarknet("yolov3-tiny.cfg", "yolov3-tiny.weights")

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]


# Set initial frame size.
frameSize = (240, 320)

# Initialize mutithreading the video stream.
# For windows use VideoCapture
vs = VideoStream(src=0, usePiCamera=True,
                 resolution=frameSize, framerate=5).start()
# Allow the camera to warm up.
time.sleep(2.0)
# camera = PiCamera()


# initialize
frame_count = 0
start = time.time()
first = True
frames = []
colour = (255, 255, 255)

while True:

    # Capture frame-by-frameq
    frame = vs.read()
    frame_count += 1
    frame = cv2.flip(frame, 1)
    frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
    frames.append(frame)

    key = cv2.waitKey(1)

    # use this only if you want to see the initial video capture
    cv2.imshow('video', frame)
    if frame_count % 500 == 0:
        end = time.time()
        # grab the frame dimensions and convert it to a blob
        (H, W) = frame.shape[:2]
        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        # You can use (288,288) to increase fps
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []
        centers = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > 0.5:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                    centers.append((centerX, centerY))

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
        texts = "scanning"

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # find positions
                centerX, centerY = centers[i][0], centers[i][1]

                if centerX <= W/3:
                    W_pos = "Move right "
                elif centerX <= (W/3 * 2):
                    W_pos = "Move right or left "
                else:
                    W_pos = "Move left "
                Addtxt = " identified"

                texts = LABELS[classIDs[i]] + Addtxt + W_pos

        # if boxes:
        #     cv2.rectangle(frame, (boxes[-1][0], boxes[-1][1]), (boxes[-1]
        #                   [0] + boxes[-1][2], boxes[-1][1] + boxes[-1][3]), colour)
        #     cv2.putText(frame, LABELS[classIDs[-1]], (boxes[-1][0],
        #                 boxes[-1][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour)
        # cv2.imshow('video', frame)

        print(texts)
        myobj = gTTS(text=texts, lang='en', slow=False)

        # Saving the converted audio in a mp3 file named audio
        myobj.save("audio.mp3")

        # Playing the converted file
        # mpg123 is a local cmd player
        os.system("mpg123 audio.mp3")
        frames[:] = []

os.remove("audio.mp3")

cap.release()
cv2.destroyAllWindows()
