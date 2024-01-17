import numpy as np
import cv2
import threading
import queue

cv2.setNumThreads(6)
framesQ = queue.Queue()

# Capturing video through webcam
webcam = cv2.VideoCapture(0)


# Define a function for detecting a specific color in a separate thread
def detect_color(lower_range, upper_range, color_name, frame):
    while True:
        # Reading the video from the webcam in image frames
        imageFrame = frame.get()

        # Convert the imageFrame from BGR(RGB color space) to HSV(hue-saturation-value) color space
        hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

        # Define mask using the lower and upper range for the specified color
        color_mask = cv2.inRange(hsvFrame, lower_range, upper_range)

        # Morphological Transform, Dilation
        # Dilate the mask for smoothing and removing small unwanted areas
        kernel = np.ones((5, 5), "uint8")
        color_mask = cv2.dilate(color_mask, kernel)

        # Applying bitwise_and operator between imageFrame and mask to detect only that particular color
        res = cv2.bitwise_and(imageFrame, imageFrame, mask=color_mask)

        # Creating contour to track the specified color
        contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

class ThreadWithReturnValue(threading.Thread):

    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                        **self._kwargs)

    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return


def readWebcam(a, quf):
    while (1):
        _, imageFr = webcam.read()
        quf.put(imageFr)
        quf.put(imageFr)
        quf.put(imageFr)
        quf.put(imageFr)
        quf.put(imageFr)


def drawFrame(contour_red, contour_green, contour_blue, contour_yellow, original_frame):
    while (1):
        imageFrame = original_frame.get()
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


# Creating separate threads for detecting different colors
red_thread = ThreadWithReturnValue(target=detect_color, args=(
    np.array([165, 100, 100], np.uint8), np.array([185, 255, 255], np.uint8), "Red Color", framesQ))
green_thread = ThreadWithReturnValue(target=detect_color, args=(
    np.array([70, 100, 100], np.uint8), np.array([85, 255, 255], np.uint8), "Green Color", framesQ))
blue_thread = ThreadWithReturnValue(target=detect_color, args=(
    np.array([105, 100, 100], np.uint8), np.array([120, 255, 255], np.uint8), "Blue Color", framesQ))
yellow_thread = ThreadWithReturnValue(target=detect_color, args=(
    np.array([35, 100, 100], np.uint8), np.array([43, 255, 255], np.uint8), "Yellow Color", framesQ))
webcam_thread = ThreadWithReturnValue(target=readWebcam, args=(
    "Red Color", framesQ))


webcam_thread.start()
red_thread.start()
green_thread.start()
blue_thread.start()
yellow_thread.start()

print("test")

webcam_thread.join()
print("test2")

drawframe_thread = ThreadWithReturnValue(target=drawFrame, args=(
    red_thread.join(), green_thread.join(), blue_thread.join(), yellow_thread.join(), framesQ))
# Starting the threads
print("test3")

drawframe_thread.start()
# Waiting for the threads to finish
print("test4")

drawframe_thread.join()
print("test5")

# Releasing the resources and closing all windows
webcam.release()
cv2.destroyAllWindows()
