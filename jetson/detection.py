#!/usr/bin/env python3
import RPi.GPIO as GPIO
import time
import sys
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log

output_pin = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.LOW)

input_source = 'csi://0'
output_source = '' 

network = "ssd-mobilenet-v2"
overlay = "box,labels,conf"
threshold = 0.5

input = videoSource(input_source)
output = videoOutput(output_source)
net = detectNet(network, threshold=threshold)

def main_loop():
    while True:
        img = input.Capture()
        if img is None:  
            continue
        detections = net.Detect(img, overlay=overlay)
        handle_detections(detections)
        output.Render(img)
        output.SetStatus("{:s} | Network {:.0f} FPS".format(network, net.GetNetworkFPS()))
        net.PrintProfilerTimes()
        if not input.IsStreaming() or not output.IsStreaming():
            break

def handle_detections(detections):
    print("detected {:d} objects in image".format(len(detections)))
    bird_present = any(net.GetClassDesc(d.ClassID).lower() == 'bird' for d in detections)
    GPIO.output(output_pin, GPIO.HIGH if bird_present else GPIO.LOW)

def user_interface():
    running = False
    while True:
        command = input("Enter command (start/stop/pause/exit): ").lower()
        if command == "start" and not running:
            running = True
            main_loop()
        elif command == "stop":
            running = False
            GPIO.cleanup()
        elif command == "pause":
            running = False
        elif command == "exit":
            GPIO.cleanup()
            sys.exit()
        else:
            print("Invalid command")

if __name__ == "__main__":
    user_interface()
