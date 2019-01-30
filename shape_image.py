# coding: utf-8

import sys
import cv2
import dlib
import numpy as np
from imutils import face_utils
import pandas as pd

face_landmark_path = 'shape_predictor_68_face_landmarks.dat'

K = [2000, 0.0, 960,
     0.0, 2000, 540,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],    #0 eyebrow_start
                         [1.330353, 7.122144, 6.903745],    #1 eyebrow_end
                         [-1.330353, 7.122144, 6.903745],   #2 eyebrow_end
                         [-6.825897, 6.760612, 4.402142],   #3 eyebrow_start
                         [5.311432, 5.485328, 3.987654],    #4 eye_start
                         [1.789930, 5.393625, 4.413414],    #5 eye_end
                         [-1.789930, 5.393625, 4.413414],   #6 eye_end
                         [-5.311432, 5.485328, 3.987654],   #7 eye_start
                         [2.005628, 1.409845, 6.165652],    #8 nose_start
                         [-2.005628, 1.409845, 6.165652],   #9 nose_end
                         [2.774015, -2.080775, 5.048531],  #10 lip_start
                         [-2.774015, -2.080775, 5.048531], #11 lip_end
                         [0.000000, -3.116408, 6.097667],  #12 lip_uppper
                         [0.000000, -7.415691, 4.070434]]) #13 lip_lower

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    #print(pose_mat)
    cameraMatrix, rotMatrix, TRANS_VEC, rotMatrixX, rotMatrixY, rotMatrixZ, EULER_ANGLE = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, EULER_ANGLE, translation_vec

#====================#Homography Transformation of face#====================#

def HT_head_pose(frame,shape): 
    reproject_obj_pts = np.float32([[object_pts[0][0],object_pts[0][1]],
                                    [object_pts[1][0],object_pts[1][1]],
                                    [object_pts[2][0],object_pts[2][1]],
                                    [object_pts[3][0],object_pts[3][1]],
                                    [object_pts[4][0],object_pts[4][1]],
                                    [object_pts[5][0],object_pts[5][1]],
                                    [object_pts[6][0],object_pts[6][1]],
                                    [object_pts[7][0],object_pts[7][1]],
                                    [object_pts[8][0],object_pts[8][1]],
                                    [object_pts[9][0],object_pts[9][1]],
                                    [object_pts[10][0],object_pts[10][1]],
                                    [object_pts[11][0],object_pts[11][1]],
                                    [object_pts[12][0],object_pts[12][1]],
                                    [object_pts[13][0],object_pts[13][1]]])  
    reproject_obj_pts = reproject_obj_pts * -30
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    #homography_mat,_ = cv2.findHomography(reproject_obj_pts, image_pts, 0, 3)
    homography_mat,_ = cv2.findHomography(image_pts, reproject_obj_pts, 0, 3)
    proj_point = cv2.perspectiveTransform(np.array([shape.astype('float32')]), homography_mat)
    proj_point = tuple(map(tuple, proj_point.reshape(68, 2)))

    #====================set white image=====================#
    rows = 1024
    cols = 1024
    img = np.zeros((rows, cols, 3), np.uint8)
    img[:,:,:] = [255,255,255]

    for (xs, ys),(xp, yp) in zip(shape, proj_point):
        cv2.circle(img, (int(xp) + 320, int(yp) + 240), 3, (0, 0, 0), -1)
        cv2.circle(img, (xs + 512, ys + 512), 1, (0, 0, 0), -1)
    
    cv2.imshow("test", img)
    #====================end====================#
    return proj_point
#====================#               end               #====================#

def main():
    # return
    args = sys.argv
    cap = cv2.imread(args[1])

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_landmark_path)

    frame = cap
    face_rects = detector(frame, 0)

    #顔が検出されたら実行
    if len(face_rects) > 0:
        shape = predictor(frame, face_rects[0])
        shape = face_utils.shape_to_np(shape)

        reprojectdst, euler_angle, translation_vec = get_head_pose(shape)

        for (x, y) in shape:
            cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)

        for start, end in line_pairs:
            cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255),8)

        cv2.putText(frame, "   =angles= =Trans=", 
                    (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), thickness=2)

        cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), thickness=2)
        cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), thickness=2)
        cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), thickness=2)

        cv2.putText(frame, "{:7.2f}".format(translation_vec[0, 0]), (160, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), thickness=2)
        cv2.putText(frame, "{:7.2f}".format(translation_vec[1, 0]), (160, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), thickness=2)
        cv2.putText(frame, "{:7.2f}".format(translation_vec[2, 0]), (160, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), thickness=2)      

    cv2.imshow("demo", cv2.resize(frame,(960, 540)))
    cv2.imwrite("../../12.png",frame)
    cv2.waitKey(50000)
            
    print("end")
    #cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
