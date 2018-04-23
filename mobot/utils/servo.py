#!/usr/bin/env python
import PCA9685 as servo
import time                  # Import necessary modules

MinPulse = 150
MaxPulse = 600

def setup():
    global pwm
    pwm = servo.PWM()

def servo_test():
    """pwm.write(0, 0, 400)
    pwm.write(14, 0, 400)
    pwm.write(15, 0, 400)

    time.sleep(2)
"""
    
    for value in range(MinPulse, MaxPulse, 5):
        pwm.write(0, 0, value)
        #pwm.write(14, 0, value)
        #pwm.write(15, 0, value)
        time.sleep(0.002)
    time.sleep(1)

    for value in range(MaxPulse, MinPulse, -5):
        pwm.write(0, 0, value)
        #pwm.write(14, 0, value)
        #pwm.write(15, 0, value)
        time.sleep(0.002)
    time.sleep(1)
    

    pwm.write(0, 0, 0)
    #pwm.write(14, 0, 0)
    #pwm.write(15, 0, 0)

 
if __name__ == '__main__':
    setup()
    servo_test()
