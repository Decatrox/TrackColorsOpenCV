from time import sleep

import numpy as np
import cv2

image = cv2.imread('coloredGlove.png')
# cv2.imshow('image', image)
# sleep(3000)
blue = image[80, 105]
red = image[190, 280]   #[158 151 142]
yellow = image[320, 173]
green = image[270, 460]
print(red, green, blue, yellow)

#[158 151 142] [460, 270] [186 103  88] [320, 340]