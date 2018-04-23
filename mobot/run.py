import sys
sys.path.append('./utils/')

from imutils.video.pivideostream import PiVideoStream
import cv2
import signal
from for_race import *

def sigint_handler(signal, frame):
    vs.stop()
    cv2.destroyAllWindows()
    setMotorSpeed(0)
    setSteeringAngle(0)
    pwm.write(0,0,0)
    sys.exit(0)

kd = 0.0
dFeedback = 0.0
if __name__ == '__main__':
    DEADZONE = 0.0  #10.0 / 180 * np.pi
    signal.signal(signal.SIGINT, sigint_handler)
    vs = PiVideoStream(resolution=(320,240), framerate=32).start()
    pwm = init()
    time.sleep(2.0)

    angle = 0.0
    lastAngle = None
    while True:
        frame = vs.read()
        print(frame.shape)

        try:
            new_angle = process_frame(frame, TEST=False)
        except:
            new_angle = -0.3*angle

        if new_angle != None:   angle = new_angle
        print("Steering angle %f" % angle)
        setMotorSpeed(40)
        
        if abs(angle) < DEADZONE:
            angle = 0

        if(lastAngle != None):
            dError = lastAngle-angle
            dFeedback = -kd*dError
        angle = min(np.pi,max(-np.pi, angle + dFeedback))


        setSteeringAngle(angle ** 3)
        print(angle)
        lastAngle = angle
        time.sleep(0.1)
