import sys
sys.path.append('./utils/')

from imutils.video.pivideostream import PiVideoStream
import time
import cv2
import signal
from for_race import *
import numpy as np

def sigint_handler(signal, frame):
    vs.stop()
    cv2.destroyAllWindows()
    setMotorSpeed(0)
    setSteeringAngle(0)
    pwm.write(0,0,0)
    sys.exit(0)

if __name__ == '__main__':
    DEADZONE = 10.0 / 180 * np.pi
    signal.signal(signal.SIGINT, sigint_handler)
    vs = PiVideoStream(resolution=(160,128), framerate=32).start()
    pwm = init()
    time.sleep(2.0)

    angle = 0.0
    while True:
        frame = vs.read()
        # s = time.time()
        # new_angle = process_frame(frame, TEST=False)
        # if new_angle != None:   angle = new_angle
        # print("Steering angle %f" % angle)
        # setMotorSpeed(40)
        # if abs(angle) < DEADZONE:
        #     angle = 0
        # setSteeringAngle(angle ** 3)
        # e = time.time()
        # print(1.0 / (e - s))
        cv2.imshow('frame2', frame)
        cv2.imshow('frame1', frame)
        time.sleep(0.1)
