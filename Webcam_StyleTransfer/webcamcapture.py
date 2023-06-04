#!/usr/bin/env python

import numpy as np
import numba
from numba import jit
import cv2 as cv
cv.OPENCV_VIDEOIO_DEBUG=1 
import time
import sys
import importlib
import os
import tensorflow as tf
import functools
import tensorflow_hub as hub

# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

print("Python version:", sys.version)
print("Numba version:", numba.__version__)
print("Numpy version:", np.__version__)
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

@jit
def applystyleexchange(model, frame, styleimg):
    return model(frame_tf, style_img_tf )[0]

#content_image = load_image('profile.jfif')
style_image = load_image('picasso.png')#'starrynight.png')#'monet.jpeg')

print("============= CAMERA FEED is now being acessed ============= ")


#Getting the Camera Feed
i = 0
found = False
for i in range(10):
        cap = cv.VideoCapture(0)#, cv.CAP_DSHOW)
        if not cap:
            print("UNABLE TO CAPTURE CAMERA")
        else:
            found = True
            print("taken camera from index: ", i)
            break

if found == False:
    print("!!! No camera was found.")
    sys.exit()

codec = 0x47504A4D  # MJPG
cap.set(cv.CAP_PROP_FPS, 20.0)
cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc('m','j','p','g'))
cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc('M','J','P','G'))
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 360)

for v in [k for k in cv.__dict__.keys() if k.startswith("CAP_PROP")]:
    print(f"cv.{v} : " , cap.get(getattr(cv, v)))


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here

    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame_tf = tf.constant(frame)
    frame_tf = frame_tf[tf.newaxis, :]
    style_img_tf = tf.constant(style_image)
    stylized_image = applystyleexchange(model, frame_tf, style_img_tf)
    
    # Display the resulting frame
    cv.imshow('frame', np.squeeze(stylized_image))
    if cv.waitKey(5) == ord('q'):
        break


    
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()


