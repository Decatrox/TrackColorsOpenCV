# import queue
#
#
# x = [0, 1, 2]
#
# def f():
#     x[1] = 27
#
# print(x)
# f()
# print(x)
#
# testQ = queue.Queue()
# testQ.put(27)
# testQ.put(29)
#
# print(testQ.get())
# print(testQ.get())
#


#shell path: wsl.exe --distribution Ubuntu

# For tf to detect the GPU:
# conda activate tf
# CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib

#check if cameras are there
# ls -al /dev/video*

#import tensorflow as tf
#tf.config.list_physical_devices('GPU')

# sudo /home/decatrox/miniconda3/envs/tf/bin/python

#To list connected cameras: lsusb | grep -i camera

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

tf.debugging.set_log_device_placement(True)

# Create some tensors
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)

print(c)

# import cv2
# index = 0
# arr = []
# while True:
#     cap = cv2.VideoCapture(index)
#     if not cap.read()[0]:
#         break
#     else:
#         arr.append(index)
#     cap.release()
#     index += 1
# print(arr)

