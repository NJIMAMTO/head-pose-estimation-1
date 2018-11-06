import cv2
import dlib
import numpy as np
from imutils import face_utils

#import pandas as pd

face_landmark_path = 'shape_predictor_68_face_landmarks.dat'

K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

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
    rotation_vec = np.float32([0,0,0])
    translation_vec = np.float32([0,0,0])
    
    #reproject_obj_pts, _  = cv2.projectPoints(object_pts, rotation_vec, translation_vec, cam_matrix,dist_coeffs)
    #reproject_obj_pts = tuple(map(tuple, reproject_obj_pts.reshape(14, 2)))

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
    reproject_obj_pts = reproject_obj_pts * 50

    pts_obj = np.float32([reproject_obj_pts[9], reproject_obj_pts[8], reproject_obj_pts[6], reproject_obj_pts[5]])
    pts_shape = np.float32([shape[39], shape[42], shape[31], shape[35]])
    homography_mat = cv2.getPerspectiveTransform(pts_obj,pts_shape)

    rows = 1024
    cols = 1024
    img = np.zeros((rows, cols, 3), np.uint8)
    img[:,:,:] = [255,255,255]

    for (x, y) in shape:
        point = np.array([[x],
                          [y],
                          [1]])
        proj_point = np.dot(np.linalg.inv(homography_mat), point)
        cv2.circle(img, (proj_point[0] + 512, proj_point[1] + 512), 1, (0, 0, 0), -1)
    
    cv2.imshow("test", img)
    return 0
#====================#               end               #====================#

def main():
    # return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to connect to camera.")
        return
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_landmark_path)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            face_rects = detector(frame, 0)

            if len(face_rects) > 0:
                shape = predictor(frame, face_rects[0])
                shape = face_utils.shape_to_np(shape)

                reprojectdst, euler_angle, trans_Vec = get_head_pose(shape)
                HT_head_pose(frame,shape)

                for i, (x, y) in enumerate(shape):
                    if i in { 17, 19, 21, 22, 24, 26, 48, 51, 54, 57, 62, 66}:
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                        #(x,y) outputed files

                    elif i in {39, 42, 27, 33}: #kizyun ziku
                        cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

                    else:
                        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

                #print(trans_Vec[0] / 1000)
                
                for start, end in line_pairs:
                    cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))
                """
                cv2.putText(frame, trans_Vec[0], (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (255, 255, 255), thickness=2)
                cv2.putText(frame, trans_Vec[1], (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (255, 255, 255), thickness=2)
                cv2.putText(frame, trans_Vec[2], (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (255, 255, 255), thickness=2)
                """
                #====================euler_angle[0, 0] -> euler_angle[2, 0] outputed files ====================#
                cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (255, 255, 255), thickness=2)
                cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (255, 255, 255), thickness=2)
                cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (255, 255, 255), thickness=2)
                
            cv2.imshow("demo", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    main()
