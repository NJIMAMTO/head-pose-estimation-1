import cv2
import dlib
import numpy as np
from imutils import face_utils
import pandas as pd

#import pandas as pd

face_landmark_path = 'shape_predictor_68_face_landmarks.dat'

K = [6.5308391993466671e+002, 0.0, 320,
     0.0, 6.5308391993466671e+002, 240,
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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to connect to camera.")
        return
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_landmark_path)

    #Save_file
    cols=["left_eyebrow_1x", "left_eyebrow_1y", "left_eyebrow_2x", "left_eyebrow_2y", "left_eyebrow_3x", "left_eyebrow_3y",
          "right_eyebrow_1x", "right_eyebrow_1y", "right_eyebrow_2x", "right_eyebrow_2y", "right_eyebrow_3x", "right_eyebrow_3y",
          "left_captain_x", "left_captain_y", "right_captain_x", "right_captain_y",
          "left_lip_x", "left_lip_y", "lip_center_upper_x", "lip_center_upper_y", "right_lip_x", "right_lip_y", "lip_center_lower_x", "lip_center_lower_y",
          "mouth_upper_x", "mouth_upper_y", "mouth_lower_x", "mouth_lower_y",
             
          "trans_x", "trans_y", "trans_z",
          "rot_x", "rot_y", "rot_z"] 
    data_frame = pd.DataFrame(index=[], columns=cols)
    tmp_list = np.array([])

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            face_rects = detector(frame, 0)

            if len(face_rects) > 0:
                shape = predictor(frame, face_rects[0])
                shape = face_utils.shape_to_np(shape)

                reprojectdst, euler_angle, trans_Vec = get_head_pose(shape)
                HT_shape = HT_head_pose(frame,shape)

                """
                for i, (x, y) in enumerate(shape):
                    if i in { 17, 19, 21, 22, 24, 26, 38, 43, 48, 51, 54, 57, 62, 66}:
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                        #(x,y) outputed files

                    elif i in {39, 42, 27, 33}: #kizyun ziku
                        cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

                    else:
                        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                """
                for i,(x, y) in enumerate(HT_shape):
                    if i in { 17, 19, 21, 22, 24, 26, 38, 43, 48, 51, 54, 57, 62, 66}:
                        tmp_list = np.append(tmp_list, x)
                        tmp_list = np.append(tmp_list, y)

                tmp_list = np.append(tmp_list, trans_Vec[0] / 100)
                tmp_list = np.append(tmp_list, trans_Vec[1] / 100)
                tmp_list = np.append(tmp_list, trans_Vec[2] / 100)

                tmp_list = np.append(tmp_list, euler_angle[0, 0])
                tmp_list = np.append(tmp_list, euler_angle[1, 0])
                tmp_list = np.append(tmp_list, euler_angle[2, 0])

                data_frame = pd.DataFrame(tmp_list, columns=data_frame.columns)
                print(data_frame)
                
            cv2.imshow("demo", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    main()
