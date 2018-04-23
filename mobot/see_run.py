# import sys
# sys.path.append('./utils/')

# import cv2
# import signal
# from for_race import *
# import numpy as np
# from picamera.array import PiRGBArray
# from picamera import PiCamera

# def sigint_handler(signal, frame):
#     vs.stop()
#     cv2.destroyAllWindows()
#     setMotorSpeed(0)
#     setSteeringAngle(0)
#     pwm.write(0,0,0)
#     sys.exit(0)

# kd = 0.0

# if __name__ == '__main__':
#     DEADZONE = 0.0  #10.0 / 180 * np.pi
#     signal.signal(signal.SIGINT, sigint_handler)

#     camera = PiCamera()
#     camera.resolution = (1080, 920) # (640, 480)
#     camera.framerate = 32
#     rawCapture = PiRGBArray(camera, size=camera.resolution)

#     pwm = init()
#     time.sleep(2.0)

#     angle = 0.0
#     lastAngle = None
#     dFeedback = 0.0
#     for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
#         frame = frame.array

#         new_angle = process_frame(frame, TEST=True)
#         if new_angle != None:   angle = new_angle
#         print("Steering angle %f" % angle)
#         setMotorSpeed(40)
        
#         if abs(angle) < DEADZONE:
#             angle = 0

#         if(lastAngle != None):
#             dError = lastAngle-angle
#             dFeedback = -kd*dError
#         angle = min(np.pi,max(-np.pi, angle + dFeedback))

#         setSteeringAngle(angle ** 3)
#         print(angle)
#         lastAngle = angle

#         rawCapture.truncate(0)
#         time.sleep(0.1)






# -*- coding: utf-8 -*-
import sys
sys.path.append('utils/')
import time
import os
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import numpy as np
import io
import motor
import PCA9685 as servo
import signal

DEADZONE = 10.0 / 180 * np.pi

# When everything is done, release the capture
def init():
    motor.setup()
    motor.ctrl(1)
    global pwm
    pwm = servo.PWM()

 
def setMotorSpeed(pwr):
    #pwr is 0 to 100 
    motor.setSpeed(pwr)

def setSteeringAngle(angle):
    #angle is -pi/2 to pi/2
    servo_min = 300.0
    servo_max = 600.0
    value = ((angle + np.pi/2) * (servo_max-servo_min)) / np.pi + servo_min
    pwm.write(0,0, int(value))

def sigint_handler(signal, frame):
    setMotorSpeed(0)
    setSteeringAngle(0)
    pwm.write(0,0,0)
    sys.exit(0)

 
if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)

    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (160, 128)
    camera.framerate = 2
    rawCapture = PiRGBArray(camera, size=camera.resolution)
     
    # allow the camera to warmup
    time.sleep(6)
    init()
    angle = 0

    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        frame = frame.array
        frame = cv2.flip(frame, 0)
        frame = cv2.flip(frame, 1)

        time.sleep(0.1)
        
        #Convert to Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #Blur image to reduce noise. if Kernel_size is bigger the image will be more blurry
        
        # change it according to your need !
        sensitivity = 75 # 30
        lower_white = np.array([0,0,255-sensitivity])
        upper_white = np.array([255,sensitivity,255])

        # Threshold the HSV image to get only white colors
        mask = cv2.inRange(gray, lower_white, upper_white)
        filtered = cv2.bitwise_and(frame, frame, mask= mask)


        Kernel_size=15
        low_threshold= 50
        high_threshold= 150
        blurred = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(filtered, (Kernel_size, Kernel_size), 0)
        blurred = cv2.dilate(blurred, np.ones((5,5),np.uint8), iterations= 3)
        blurred = cv2.GaussianBlur(blurred, (Kernel_size, Kernel_size), 0)
        blurred = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)

        #Perform canny edge-detection.
        #If a pixel gradient is higher than high_threshold is considered as an edge.
        #if a pixel gradient is lower than low_threshold is is rejected , it is not an edge.
        #Bigger high_threshold values will provoque to find less edges.
        #Canny recommended ratio upper:lower  between 2:1 or 3:1
        # edged = cv2.Canny(blurred, low_threshold, high_threshold)
        # edged = cv2.dilate(blurred, np.ones((5,5),np.uint8), iterations= 3)
        # edged = cv2.erode(edged, np.ones((5,5),np.uint8), iterations= 3)
        # edged = cv2.dilate(edged, np.ones((5,5),np.uint8), iterations= 3)
        # edged = cv2.erode(edged, np.ones((5,5),np.uint8), iterations= 3)
        # edged = cv2.dilate(edged, np.ones((5,5),np.uint8), iterations= 3)

       
        
        edged = blurred

        def get_vertices_for_img(img):
            imshape = img.shape
            height = imshape[0]
            width = imshape[1]
            height_ratio = 0.8
            width_ratio = 0.40
            # vert = np.array([[(0.5 * height, 1) , (0, 0),  (0, width), (0.5 * height, width-1)]], dtype=np.int32)
            vert = np.array([[(1,height_ratio * height) , (width_ratio * width,height),  ((1-width_ratio) * width, height), (width-1, height_ratio * height)]], dtype=np.int32)
            return vert

        def get_region_of_interest(img):
            mask = np.zeros_like(img)
            verts = get_vertices_for_img(img)
            cv2.fillPoly(mask, verts, (255, 255, 255))
            masked_img = cv2.bitwise_and(img, mask)
            return masked_img

        roi_masked = get_region_of_interest(edged)

        ret, thresholded = cv2.threshold(roi_masked, 60, 255, cv2.THRESH_BINARY)
        srcimg, contours, hierarchy = cv2.findContours(thresholded.copy(), 1, cv2.CHAIN_APPROX_NONE)

        def find_steering(img, cx,cy):
            imshape = img.shape
            height = imshape[0]
            width = imshape[1]
            return np.arctan((cx - height) / (cy - width/2 + 1e-6))

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
     
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            cv2.circle(frame,(cx,cy), 25, (0,0,255), -1)

            # cv2.line(frame,(cx,0),(cx,720),(255,0,0),1)
            # cv2.line(frame,(0,cy),(1280,cy),(255,0,0),1)
            height = frame.shape[0]
            width = frame.shape[1]
            cv2.line(frame, (round(cx), round(cy)), (round(width/2), round(height + 1e-6)), (255,0,0), 5)
            angle = find_steering(frame, cx, cy)
            cv2.putText(frame, '%f' % angle, (round(width/2), round(height)), cv2.FONT_HERSHEY_SIMPLEX, 5, (255,255,255))

            cv2.drawContours(frame, c, -1, (0,255,0), 10)


        # cv2.imshow('filtered', filtered)
        # cv2.imshow("cROI", roi_masked)
        cv2.imshow("tresholded", thresholded)
        cv2.imshow("frame", frame)

        setMotorSpeed(40)
        if abs(angle) < DEADZONE:
            angle = 0
        setSteeringAngle(angle ** 3)
        

        rawCapture.truncate(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            setMotorSpeed(0)
            break

       
       

    video_capture.release()
    cv2.destroyAllWindows()
