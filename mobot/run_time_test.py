import sys
sys.path.append('./utils/')
from imutils.video.pivideostream import PiVideoStream
import time
import cv2
import signal
from time_test import process_frame

def sigint_handler(signal, frame):
    vs.stop()
    cv2.destroyAllWindows()
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)
    vs = PiVideoStream(resolution=(320,240), framerate=32).start()
    time.sleep(2.0)
    while True:
        frame = vs.read()
        angle = process_frame(frame, TEST=False)
        print(angle)


