from keras import backend as K
import imutils
from keras.models import load_model
import numpy as np
import keras
import requests
from scipy.spatial import distance as dist
from imutils import face_utils
import time
import dlib
import cv2, os, sys
import collections
import random
import face_recognition
import pickle
import math
import threading
import tensorflow as tf


class FacialLandMarksPosition:
    left_eye_start_index, left_eye_end_index = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    right_eye_start_index, right_eye_end_index = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def predict_eye_state(model, image):
    image = cv2.resize(image, (20, 10))
    image = image.astype(dtype=np.float32)

    image_batch = np.reshape(image, (1, 10, 20, 1))
    image_batch = keras.applications.mobilenet.preprocess_input(image_batch)

    return np.argmax(model.predict(image_batch)[0])


facial_landmarks_predictor = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(facial_landmarks_predictor)

model = load_model('eyes_detection.hdf5')

cap = cv2.VideoCapture(0)
scale = 0.5
countClose = 0
currState = 0
alarmThreshold = 5


while (True):
    c = time.time()
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    original_height, original_width = image.shape[:2]
    resized_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

    lab = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB)
    l, _, _ = cv2.split(lab)
    resized_height, resized_width = l.shape[:2]
    height_ratio, width_ratio = original_height / resized_height, original_width / resized_width

    face_locations = face_recognition.face_locations(l, model='hog')

    if len(face_locations):

        top, right, bottom, left = face_locations[0]
        x1, y1, x2, y2 = left, top, right, bottom
        x1 = int(x1 * width_ratio)
        y1 = int(y1 * height_ratio)
        x2 = int(x2 * width_ratio)
        y2 = int(y2 * height_ratio)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shape = predictor(gray, dlib.rectangle(x1, y1, x2, y2))
        face_landmarks = face_utils.shape_to_np(shape)

        left_eye_indices = face_landmarks[FacialLandMarksPosition.left_eye_start_index:
                                          FacialLandMarksPosition.left_eye_end_index]

        (x, y, w, h) = cv2.boundingRect(np.array([left_eye_indices]))
        left_eye = gray[y:y + h, x:x + w]

        right_eye_indices = face_landmarks[FacialLandMarksPosition.right_eye_start_index:
                                           FacialLandMarksPosition.right_eye_end_index]

        (x, y, w, h) = cv2.boundingRect(np.array([right_eye_indices]))
        right_eye = gray[y:y + h, x:x + w]

        left_eye_open = 'yes' if predict_eye_state(model=model, image=left_eye) else 'no'
        right_eye_open = 'yes' if predict_eye_state(model=model, image=right_eye) else 'no'

        print('left eye open: {0}    right eye open: {1}'.format(left_eye_open, right_eye_open))

        if left_eye_open == 'yes' and right_eye_open == 'yes':
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            currState = 0
            countClose = 0

        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            currState = 1
            countClose +=1


    frame = cv2.flip(frame, 1)
    if countClose > alarmThreshold:
        cv2.putText(frame, "Sleep detected! Alarm!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),
                    lineType=cv2.LINE_AA)
    cv2.imshow('Sleep Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()