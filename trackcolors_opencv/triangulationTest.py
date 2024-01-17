# import cv2 as cv
# import glob
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def calibrate_camera(images_folder):
#     images_names = glob.glob(images_folder)
#     images = []
#     for imname in images_names:
#         im = cv.imread(imname, 1)
#         images.append(im)
#
#     # plt.figure(figsize = (10,10))
#     # ax = [plt.subplot(2,2,i+1) for i in range(4)]
#     #
#     # for a, frame in zip(ax, images):
#     #     a.imshow(frame[:,:,[2,1,0]])
#     #     a.set_xticklabels([])
#     #     a.set_yticklabels([])
#     # plt.subplots_adjust(wspace=0, hspace=0)
#     # plt.show()
#
#     # criteria used by checkerboard pattern detector.
#     # Change this if the code can't find the checkerboard
#     criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#
#     rows = 5  # number of checkerboard rows.
#     columns = 8  # number of checkerboard columns.
#     world_scaling = 1.  # change this to the real world square size. Or not.
#
#     # coordinates of squares in the checkerboard world space
#     objp = np.zeros((rows * columns, 3), np.float32)
#     objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
#     objp = world_scaling * objp
#
#     # frame dimensions. Frames should be the same size.
#     width = images[0].shape[1]
#     height = images[0].shape[0]
#
#     # Pixel coordinates of checkerboards
#     imgpoints = []  # 2d points in image plane.
#
#     # coordinates of the checkerboard in checkerboard world space.
#     objpoints = []  # 3d point in real world space
#
#     for frame in images:
#         gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#
#         # find the checkerboard
#         ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)
#
#         if ret == True:
#             # Convolution size used to improve corner detection. Don't make this too large.
#             conv_size = (11, 11)
#
#             # opencv can attempt to improve the checkerboard coordinates
#             corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
#             cv.drawChessboardCorners(frame, (rows, columns), corners, ret)
#             cv.imshow('img', frame)
#             cv.waitKey(500)
#
#             objpoints.append(objp)
#             imgpoints.append(corners)
#
#     ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
#     print('rmse:', ret)
#     print('camera matrix:\n', mtx)
#     print('distortion coeffs:', dist)
#     print('Rs:\n', rvecs)
#     print('Ts:\n', tvecs)
#
#     return mtx, dist
#
#
# def stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_folder):
#     # read the synched frames
#     images_names = glob.glob(frames_folder)
#     images_names = sorted(images_names)
#     c1_images_names = images_names[:len(images_names) // 2]
#     c2_images_names = images_names[len(images_names) // 2:]
#
#     c1_images = []
#     c2_images = []
#     for im1, im2 in zip(c1_images_names, c2_images_names):
#         _im = cv.imread(im1, 1)
#         c1_images.append(_im)
#
#         _im = cv.imread(im2, 1)
#         c2_images.append(_im)
#
#     # change this if stereo calibration not good.
#     criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
#
#     rows = 5  # number of checkerboard rows.
#     columns = 8  # number of checkerboard columns.
#     world_scaling = 1.  # change this to the real world square size. Or not.
#
#     # coordinates of squares in the checkerboard world space
#     objp = np.zeros((rows * columns, 3), np.float32)
#     objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
#     objp = world_scaling * objp
#
#     # frame dimensions. Frames should be the same size.
#     width = c1_images[0].shape[1]
#     height = c1_images[0].shape[0]
#
#     # Pixel coordinates of checkerboards
#     imgpoints_left = []  # 2d points in image plane.
#     imgpoints_right = []
#
#     # coordinates of the checkerboard in checkerboard world space.
#     objpoints = []  # 3d point in real world space
#
#     for frame1, frame2 in zip(c1_images, c2_images):
#         gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
#         gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
#         c_ret1, corners1 = cv.findChessboardCorners(gray1, (5, 8), None)
#         c_ret2, corners2 = cv.findChessboardCorners(gray2, (5, 8), None)
#
#         if c_ret1 == True and c_ret2 == True:
#             corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
#             corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
#
#             cv.drawChessboardCorners(frame1, (5, 8), corners1, c_ret1)
#             cv.imshow('img', frame1)
#
#             cv.drawChessboardCorners(frame2, (5, 8), corners2, c_ret2)
#             cv.imshow('img2', frame2)
#             cv.waitKey(500)
#
#             objpoints.append(objp)
#             imgpoints_left.append(corners1)
#             imgpoints_right.append(corners2)
#
#     stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
#     ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1,
#                                                                  dist1,
#                                                                  mtx2, dist2, (width, height), criteria=criteria,
#                                                                  flags=stereocalibration_flags)
#
#     print(ret)
#     return R, T
#
#
# def triangulate(mtx1, mtx2, R, T):
#     uvs1 = [[458, 86], [451, 164], [287, 181],
#             [196, 383], [297, 444], [564, 194],
#             [562, 375], [596, 520], [329, 620],
#             [488, 622], [432, 52], [489, 56]]
#
#     uvs2 = [[540, 311], [603, 359], [542, 378],
#             [525, 507], [485, 542], [691, 352],
#             [752, 488], [711, 605], [549, 651],
#             [651, 663], [526, 293], [542, 290]]
#
#     uvs1 = np.array(uvs1)
#     uvs2 = np.array(uvs2)
#
#     frame1 = cv.imread('testing/_C1.png')
#     frame2 = cv.imread('testing/_C2.png')
#
#     plt.imshow(frame1[:, :, [2, 1, 0]])
#     plt.scatter(uvs1[:, 0], uvs1[:, 1])
#     plt.show()  # this call will cause a crash if you use cv.imshow() above. Comment out cv.imshow() to see this.
#
#     plt.imshow(frame2[:, :, [2, 1, 0]])
#     plt.scatter(uvs2[:, 0], uvs2[:, 1])
#     plt.show()  # this call will cause a crash if you use cv.imshow() above. Comment out cv.imshow() to see this
#
#     # RT matrix for C1 is identity.
#     RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
#     P1 = mtx1 @ RT1  # projection matrix for C1
#
#     # RT matrix for C2 is the R and T obtained from stereo calibration.
#     RT2 = np.concatenate([R, T], axis=-1)
#     P2 = mtx2 @ RT2  # projection matrix for C2
#
#     def DLT(P1, P2, point1, point2):
#
#         A = [point1[1] * P1[2, :] - P1[1, :],
#              P1[0, :] - point1[0] * P1[2, :],
#              point2[1] * P2[2, :] - P2[1, :],
#              P2[0, :] - point2[0] * P2[2, :]
#              ]
#         A = np.array(A).reshape((4, 4))
#         # print('A: ')
#         # print(A)
#
#         B = A.transpose() @ A
#         from scipy import linalg
#         U, s, Vh = linalg.svd(B, full_matrices=False)
#
#         print('Triangulated point: ')
#         print(Vh[3, 0:3] / Vh[3, 3])
#         return Vh[3, 0:3] / Vh[3, 3]
#
#     p3ds = []
#     for uv1, uv2 in zip(uvs1, uvs2):
#         _p3d = DLT(P1, P2, uv1, uv2)
#         p3ds.append(_p3d)
#     p3ds = np.array(p3ds)
#
#     from mpl_toolkits.mplot3d import Axes3D
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_xlim3d(-15, 5)
#     ax.set_ylim3d(-10, 10)
#     ax.set_zlim3d(10, 30)
#
#     connections = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [1, 9], [2, 8], [5, 9], [8, 9],
#                    [0, 10], [0, 11]]
#     for _c in connections:
#         print(p3ds[_c[0]])
#         print(p3ds[_c[1]])
#         ax.plot(xs=[p3ds[_c[0], 0], p3ds[_c[1], 0]], ys=[p3ds[_c[0], 1], p3ds[_c[1], 1]],
#                 zs=[p3ds[_c[0], 2], p3ds[_c[1], 2]], c='red')
#     ax.set_title('This figure can be rotated.')
#     # uncomment to see the triangulated pose. This may cause a crash if youre also using cv.imshow() above.
#     plt.show()
#
#
# mtx1, dist1 = calibrate_camera(images_folder='D2/*')
# mtx2, dist2 = calibrate_camera(images_folder='J2/*')
#
# R, T = stereo_calibrate(mtx1, dist1, mtx2, dist2, 'synched/*')
#
# # this call might cause segmentation fault error. This is due to calling cv.imshow() and plt.show()
# triangulate(mtx1, mtx2, R, T)

# ------------------------------
# Notice
# ------------------------------

# Copyright 1966 Clayton Darwin claytondarwin@gmail.com

# ------------------------------
# Imports
# ------------------------------

import time
import traceback

import numpy as np
import cv2

import targeting_tools as tt


# ------------------------------
# Testing
# ------------------------------

def run():
    # ------------------------------
    # full error catch
    # ------------------------------
    try:

        # ------------------------------
        # set up cameras
        # ------------------------------

        # cameras variables
        left_camera_source = 2
        right_camera_source = 4
        pixel_width = 640
        pixel_height = 480
        angle_width = 78
        angle_height = 64  # 63
        frame_rate = 20
        camera_separation = 5 + 15 / 16

        # left camera 1
        ct1 = tt.Camera_Thread()
        ct1.camera_source = left_camera_source
        ct1.camera_width = pixel_width
        ct1.camera_height = pixel_height
        ct1.camera_frame_rate = frame_rate

        # right camera 2
        ct2 = tt.Camera_Thread()
        ct2.camera_source = right_camera_source
        ct2.camera_width = pixel_width
        ct2.camera_height = pixel_height
        ct2.camera_frame_rate = frame_rate

        # camera coding
        # ct1.camera_fourcc = cv2.VideoWriter_fourcc(*"YUYV")
        # ct2.camera_fourcc = cv2.VideoWriter_fourcc(*"YUYV")
        ct1.camera_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        ct2.camera_fourcc = cv2.VideoWriter_fourcc(*"MJPG")

        # start cameras
        ct1.start()
        ct2.start()

        # ------------------------------
        # set up angles
        # ------------------------------

        # cameras are the same, so only 1 needed
        angler = tt.Frame_Angles(pixel_width, pixel_height, angle_width, angle_height)
        angler.build_frame()

        # ------------------------------
        # set up motion detection
        # ------------------------------

        # motion camera1
        # using default detect values
        targeter1 = tt.Frame_Motion()
        targeter1.contour_min_area = 1
        targeter1.targets_max = 1
        targeter1.target_on_contour = True  # False = use box size
        targeter1.target_return_box = False  # (x,y,bx,by,bw,bh)
        targeter1.target_return_size = True  # (x,y,%frame)
        targeter1.contour_draw = True
        targeter1.contour_box_draw = False
        targeter1.targets_draw = True

        # motion camera2
        # using default detect values
        targeter2 = tt.Frame_Motion()
        targeter2.contour_min_area = 1
        targeter2.targets_max = 1
        targeter2.target_on_contour = True  # False = use box size
        targeter2.target_return_box = False  # (x,y,bx,by,bw,bh)
        targeter2.target_return_size = True  # (x,y,%frame)
        targeter2.contour_draw = True
        targeter2.contour_box_draw = False
        targeter2.targets_draw = True

        # ------------------------------
        # stabilize
        # ------------------------------

        # pause to stabilize
        time.sleep(0.5)

        # ------------------------------
        # targeting loop
        # ------------------------------

        # variables
        maxsd = 2  # maximum size difference of targets, percent of frame
        klen = 3  # length of target queues, positive target frames required to reset set X,Y,Z,D

        # target queues
        x1k, y1k, x2k, y2k = [], [], [], []
        x1m, y1m, x2m, y2m = 0, 0, 0, 0

        # last positive target
        # from camera baseline midpoint
        X, Y, Z, D = 0, 0, 0, 0

        # loop
        while 1:

            # get frames
            frame1 = ct1.next(black=True, wait=1)
            frame2 = ct2.next(black=True, wait=1)

            # motion detection targets
            targets1 = targeter1.targets(frame1)
            targets2 = targeter2.targets(frame2)

            # check 1: motion in both frames
            if not (targets1 and targets2):
                x1k, y1k, x2k, y2k = [], [], [], []  # reset
            else:

                # split
                x1, y1, s1 = targets1[0]
                x2, y2, s2 = targets2[0]

                # check 2: similar size
                # if 100*(abs(s1-s2)/max(s1,s2)) > minsd:
                if abs(s1 - s2) > maxsd:
                    x1k, y1k, x2k, y2k = [], [], [], []  # reset
                else:

                    # update queues
                    x1k.append(x1)
                    y1k.append(y1)
                    x2k.append(x2)
                    y2k.append(y2)

                    # check 3: queues full
                    if len(x1k) >= klen:
                        # trim
                        x1k = x1k[-klen:]
                        y1k = y1k[-klen:]
                        x2k = x2k[-klen:]
                        y2k = y2k[-klen:]

                        # mean values
                        x1m = sum(x1k) / klen
                        y1m = sum(y1k) / klen
                        x2m = sum(x2k) / klen
                        y2m = sum(y2k) / klen

                        # get angles from camera centers
                        xlangle, ylangle = angler.angles_from_center(x1m, y1m, top_left=True, degrees=True)
                        xrangle, yrangle = angler.angles_from_center(x2m, y2m, top_left=True, degrees=True)

                        # triangulate
                        X, Y, Z, D = angler.location(camera_separation, (xlangle, ylangle), (xrangle, yrangle),
                                                     center=True, degrees=True)

            # display camera centers
            angler.frame_add_crosshairs(frame1)
            angler.frame_add_crosshairs(frame2)

            # display coordinate data
            fps1 = int(ct1.current_frame_rate)
            fps2 = int(ct2.current_frame_rate)
            text = 'X: {:3.1f}\nY: {:3.1f}\nZ: {:3.1f}\nD: {:3.1f}\nFPS: {}/{}'.format(X, Y, Z, D, fps1, fps2)
            lineloc = 0
            lineheight = 30
            for t in text.split('\n'):
                lineloc += lineheight
                cv2.putText(frame1,
                            t,
                            (10, lineloc),  # location
                            cv2.FONT_HERSHEY_PLAIN,  # font
                            # cv2.FONT_HERSHEY_SIMPLEX, # font
                            1.5,  # size
                            (0, 255, 0),  # color
                            1,  # line width
                            cv2.LINE_AA,  #
                            False)  #

            # display current target
            if x1k:
                targeter1.frame_add_crosshairs(frame1, x1m, y1m, 48)
                targeter2.frame_add_crosshairs(frame2, x2m, y2m, 48)

                # display frame
            cv2.imshow("Left Camera 1", frame1)
            cv2.imshow("Right Camera 2", frame2)

            # detect keys
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty('Left Camera 1', cv2.WND_PROP_VISIBLE) < 1:
                break
            elif cv2.getWindowProperty('Right Camera 2', cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord('q'):
                break
            elif key != 255:
                print('KEY PRESS:', [chr(key)])

    # ------------------------------
    # full error catch
    # ------------------------------
    except:
        print(traceback.format_exc())

    # ------------------------------
    # close all
    # ------------------------------

    # close camera1
    try:
        ct1.stop()
    except:
        pass

    # close camera2
    try:
        ct2.stop()
    except:
        pass

    # kill frames
    cv2.destroyAllWindows()

    # done
    print('DONE')


# ------------------------------
# run
# ------------------------------

if __name__ == '__main__':
    run()

# ------------------------------
# end
# ------------------------------
