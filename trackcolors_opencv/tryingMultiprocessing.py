import numpy as np
import cv2
from multiprocessing import Process
from multiprocessing import Queue

#if __name__ == '__main__':
#cv2.setNumThreads(6)
framesQ = Queue()
redQ = Queue()
blueQ = Queue()
greenQ = Queue()
yellowQ = Queue()

# Capturing video through webcam
webcam = cv2.VideoCapture(0)


# Define a function for detecting a specific color in a separate thread
def detect_color(lower_range, upper_range, color_name, frame):
    while True:

        imageFrame = frame.get()

        # Convert the imageFrame from BGR(RGB color space) to HSV(hue-saturation-value) color space
        hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

        # Define mask using the lower and upper range for the specified color
        color_mask = cv2.inRange(hsvFrame, lower_range, upper_range)

        # Morphological Transform, Dilation
        # Dilate the mask for smoothing and removing small unwanted areas
        kernal = np.ones((5, 5), "uint8")
        color_mask = cv2.dilate(color_mask, kernal)

        # Applying bitwise_and operator between imageFrame and mask to detect only that particular color
        cv2.bitwise_and(imageFrame, imageFrame, mask=color_mask)

        # Creating contour to track the specified color
        contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if color_name == "Blue Color":
            blueQ.put(contours)
        elif color_name == "Red Color":
            redQ.put(contours)
        elif color_name == "Green Color":
            greenQ.put(contours)
        elif color_name == "Yellow Color":
            yellowQ.put(contours)



def readWebcam(a, quf):
    while (1):
        _, imageFr = webcam.read()
        quf.put(imageFr)
        quf.put(imageFr)
        quf.put(imageFr)
        quf.put(imageFr)
        quf.put(imageFr)


def drawFrame(qred, qgreen, qblue, qyellow, original_frame):
    while (1):
        imageFrame = original_frame.get()
        contour_red = qred.get()
        contour_green = qgreen.get()
        contour_blue = qblue.get()
        contour_yellow = qyellow.get()

        for pic, contour in enumerate(contour_red):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # change the color later
                cv2.putText(imageFrame, "Red", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255))  # chaneg color as well

        for pic, contour in enumerate(contour_green):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # change the color later
                cv2.putText(imageFrame, "Green", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 255, 0))  # chaneg color as well

        for pic, contour in enumerate(contour_blue):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # change the color later
                cv2.putText(imageFrame, "Blue", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (255, 0, 0))  # chaneg color as well

        for pic, contour in enumerate(contour_yellow):
            area = cv2.contourArea(contour)
            if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 255),
                                           2)  # change the color later
                cv2.putText(imageFrame, "Yellow", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 255, 255))  # chaneg color as well

        # Displaying the color detection results
        cv2.imshow("Color Detection", imageFrame)
        if cv2.waitKey(1) == 27:  # Exit if 'Esc' key is pressed
            break


red_thread = Process(target=detect_color, args=(
np.array([165, 100, 100], np.uint8), np.array([185, 255, 255], np.uint8), "Red Color", framesQ))
green_thread = Process(target=detect_color, args=(
np.array([70, 100, 100], np.uint8), np.array([85, 255, 255], np.uint8), "Green Color", framesQ))
blue_thread = Process(target=detect_color, args=(
np.array([105, 100, 100], np.uint8), np.array([120, 255, 255], np.uint8), "Blue Color", framesQ))
yellow_thread = Process(target=detect_color, args=(
np.array([35, 100, 100], np.uint8), np.array([43, 255, 255], np.uint8), "Yellow Color", framesQ))
webcam_thread = Process(target=readWebcam, args=(
"Red Color", framesQ))
drawframe_thread = Process(target=drawFrame, args=(
redQ, greenQ, blueQ, yellowQ, framesQ))

webcam_thread.start()
red_thread.start()
green_thread.start()
blue_thread.start()
yellow_thread.start()
drawframe_thread.start()
print("test")

webcam_thread.join()
red_thread.join()
green_thread.join()
blue_thread.join()
yellow_thread.join()
drawframe_thread.join()

webcam.release()
cv2.destroyAllWindows()
