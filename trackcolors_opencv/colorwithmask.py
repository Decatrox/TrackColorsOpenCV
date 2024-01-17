import cv2
import numpy as np

# Hue: Green(70-85), Yellow(35-43), Red(165-185), Blue(105, 120) SV are 100-255
# Hue Borders Green(75-80)
# White lower = np.array([0, 0, 168]) upper = np.array([172, 111, 255])
# lower = np.array([15, 150, 20])
# upper = np.array([35, 255, 255])
lower = np.array([35, 100, 100])
upper = np.array([43, 255, 255])

video = cv2.VideoCapture(0)
# For HSV, hue range is [0,179], saturation range is [0,255], and value range is [0,255].
# Yellow 35-43 49-60 75-85
# Green 50 11 58
while True:
    success, img = video.read()
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image, lower, upper)

    cv2.imshow("mask", mask)
    cv2.imshow("webcam", img)

    cv2.waitKey(1)
