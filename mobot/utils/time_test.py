# -*- coding: utf-8 -*-
import sys
import time
import os
import cv2
import numpy as np
import io
import signal

DEADZONE = 10.0 / 180 * np.pi

# When everything is done, release the capture
def process_frame(frame, TEST=False):
    times = dict()
    start = time.time()
    frame = cv2.flip(frame, 0)
    frame = cv2.flip(frame, 1)

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
    blurred = cv2.GaussianBlur(blurred, (Kernel_size, Kernel_size), 0)
    # blurred = cv2.dilate(blurred, np.ones((5,5),np.uint8), iterations= 3)
    # blurred = cv2.GaussianBlur(blurred, (Kernel_size, Kernel_size), 0)
    # blurred = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)

    def get_vertices_for_img(img):
        imshape = img.shape
        height = imshape[0]
        width = imshape[1]
        height_ratio = 0.8
        width_ratio = 0.40
        vert = np.array([[(1,height_ratio * height) , (width_ratio * width,height),  ((1-width_ratio) * width, height), (width-1, height_ratio * height)]], dtype=np.int32)
        return vert

    def get_region_of_interest(img):
        mask = np.zeros_like(img)
        verts = get_vertices_for_img(img)
        cv2.fillPoly(mask, verts, (255, 255, 255))
        masked_img = cv2.bitwise_and(img, mask)
        return masked_img

    roi_masked = get_region_of_interest(blurred)
    ret, thresholded = cv2.threshold(roi_masked, 60, 255, cv2.THRESH_BINARY)
    end = time.time()
    times['everything'] = times.get('everything', []) + [end-start]

    start = time.time()
    srcimg, contours, hierarchy = cv2.findContours(thresholded.copy(), 1, cv2.CHAIN_APPROX_NONE)
    end = time.time()
    times['contour'] = times.get('contour', []) + [end-start]

    def find_steering(img, cx,cy):
        imshape = img.shape
        height = imshape[0]
        width = imshape[1]
        return np.arctan((cx - height) / (cy - width/2 + 1e-6))

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
 
        cx = int(M['m10']/(M['m00'] + 1e-6))
        cy = int(M['m01']/(M['m00'] + 1e-6))

        if TEST:    cv2.circle(frame,(cx,cy), 25, (0,0,255), -1)
        height = frame.shape[0]
        width = frame.shape[1]
        if TEST:    cv2.line(frame, (round(cx), round(cy)), (round(width/2), round(height + 1e-6)), (255,0,0), 5)
        angle = find_steering(frame, cx, cy)

        if TEST:    cv2.drawContours(frame, c, -1, (0,255,0), 10)

    if TEST:
        cv2.imshow('filtered', filtered)
        cv2.imshow("cROI", roi_masked)
        cv2.imshow("tresholded", thresholded)
        cv2.imshow("frame", frame)
   
    print("Everything else: %f; contour time: %f" % (times['everything'][-1], times['contour'][-1]))
