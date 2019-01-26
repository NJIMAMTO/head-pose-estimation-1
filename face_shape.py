# coding: utf-8

import sys
import cv2
import dlib
import numpy as np
from imutils import face_utils
import pandas as pd

face_landmark_path = 'shape_predictor_68_face_landmarks.dat'
"""
#Webcam
K = [6.5308391993466671e+002, 0.0, 320,
     0.0, 6.5308391993466671e+002, 240,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]
"""
#3Dface
K = [3888.0, 0.0, 600,
     0.0, 3888.0, 600,
     0.0, 0.0, 1.0]
D = [0, 0, 0.0, 0.0, 0]

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

class Head_Pose:
    def __init__(self,shape):
        
        self.image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])
        self.shape = shape

    def get(self):
        _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, self.image_pts, cam_matrix, dist_coeffs)

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

    def Homography_Trans(self,frame): 
        #objct_ptsからxy座標のみを抽出
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

        homography_mat,_ = cv2.findHomography(self.image_pts, reproject_obj_pts, 0, 3)
        proj_point = cv2.perspectiveTransform(np.array([self.shape.astype('float32')]), homography_mat)
        proj_point = tuple(map(tuple, proj_point.reshape(68, 2)))

        #====================set white image=====================#
        rows = 1200
        cols = 1200
        img = np.zeros((rows, cols, 3), np.uint8)
        img[:,:,:] = [255,255,255]

        for (xs, ys),(xp, yp) in zip(self.shape, proj_point):
            cv2.circle(img, (int(xp) + 320, int(yp) + 240), 3, (0, 0, 0), -1)
            cv2.circle(img, (xs + 512, ys + 512), 1, (0, 0, 0), -1)
        
        cv2.imshow("test", img)
        #====================end====================#
        return proj_point
    #====================#               end               #====================#

def main():
    """
    args = sys.argv
    cap = cv2.VideoCapture(args[1])
    #cap = cv2.VideoCapture(0)
    """

    path = "/media/mokugyo/ボリューム/3Dface"
    folder = ["F_Angry","F_Disgust","F_Fear","F_Happy","F_Neutral","F_Surprise","F_Unhappy",
        "M_Angry","M_Disgust","M_Fear","M_Happy","M_Neutral","M_Surprise","M_Unhappy"]
    V_S = ["V0S","V2L","V4L"]

    for ii in range(0,14):
        for xx in range(0, 3):
            file_path = path + "/" + folder[ii] + "/" + V_S[xx]
            print(file_path)
            cap = cv2.VideoCapture(file_path + '.mp4')

            if not cap.isOpened():
                print("Unable to connect to camera.")
                return
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(face_landmark_path)

            #Save_file (feature points -> https://cloud.githubusercontent.com/assets/16308037/24229391/1910e9cc-0fb4-11e7-987b-0fecce2c829e.JPG)
            cols = ["18-37","20-37","22-37","23-46","25-46","27-46","49-34","52-34","55-34","58-34",
                    "trans_x", "trans_y", "trans_z",
                    "rot_x", "rot_y", "rot_z"] 
            data_frame = pd.DataFrame(index=[], columns=cols)

            while cap.isOpened():
                ret, im = cap.read()
                if ret:
                    frame = im
                    ########################
                    #frame = im[960:1980,0:540]
                    #frame = im[540:1080,0:960]
                    ########################
                    face_rects = detector(frame, 0)
                    tmp_list = np.array([])

                    #顔が検出されたら実行
                    if len(face_rects) > 0:
                        shape = predictor(frame, face_rects[0])
                        shape = face_utils.shape_to_np(shape)

                        HP = Head_Pose(shape)
                        _ , euler_angle, trans_Vec = HP.get()
                        HT_shape = HP.Homography_Trans(frame)

                        Reference_point = np.float32([HT_shape[36], HT_shape[45], HT_shape[33]])
                        for i in range(68):
                            if i in { 17, 19, 21}:
                                norm = np.linalg.norm(HT_shape[i] - Reference_point[0], ord=2)
                                tmp_list = np.append(tmp_list, norm)
                            elif i in {22, 24, 26}:
                                norm = np.linalg.norm(HT_shape[i] - Reference_point[1], ord=2)
                                tmp_list = np.append(tmp_list, norm)
                            elif i in {48, 51, 54, 57}:
                                norm = np.linalg.norm(HT_shape[i] - Reference_point[2], ord=2)
                                tmp_list = np.append(tmp_list, norm)

                        tmp_list = np.append(tmp_list, [trans_Vec[0]/100, trans_Vec[1]/100, trans_Vec[1]/100])
                        tmp_list = np.append(tmp_list, [euler_angle[0, 0], euler_angle[1, 0], euler_angle[2, 0]])
                                
                        df = pd.Series(tmp_list, index = data_frame.columns)
                        data_frame = data_frame.append(df,ignore_index = True) 

                    #検出されなかった場合 data_frameに空フレームを追加
                    else:
                        tmp_list = np.zeros([16])
                        tmp_list[:] = np.nan
                        df = pd.Series(tmp_list, index = data_frame.columns)
                        data_frame = data_frame.append(df,ignore_index = True)    

                    cv2.imshow("demo", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
                            
            print("end")
            cap.release()
            cv2.destroyAllWindows()
            #data_frame.to_csv(args[2])
            data_frame.to_csv(file_path + '.csv')

if __name__ == '__main__':
    main()
