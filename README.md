# head-pose-estimation

表情・頭部姿勢変化を使った感情認識プログラムの流れ：**顔画像・映像入力->特徴点検出**->前処理->機械学習

#### 研究室メンバーへ
ここでは映像入力から前処理までを実装しています。
これは以下で説明されているものを改造したものです（主にCSVへの出力部分を追加）．

>forked from https://github.com/lincolnhard/head-pose-estimation

Real-time head pose estimation built with OpenCV and dlib 

<b>2D:</b><br>Using dlib for facial features tracking, modified from http://dlib.net/webcam_face_pose_ex.cpp.html
<br>The algorithm behind it is described in http://www.csc.kth.se/~vahidk/papers/KazemiCVPR14.pdf
<br>It applies cascaded regression trees to predict shape(feature locations) change in every frame.
<br>Splitting nodes of trees are trained in random, greedy, maximizing variance reduction fashion.
<br>The well trained model can be downloaded from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 
<br>Training set is based on i-bug 300-W datasets. It's annotation is shown below:<br><br>
![ibug](https://cloud.githubusercontent.com/assets/16308037/24229391/1910e9cc-0fb4-11e7-987b-0fecce2c829e.JPG)
<br><br>
<b>3D:</b><br>To match with 2D image points(facial features) we need their corresponding 3D model points. 
<br>http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp provides a similar 3D facial feature model.
<br>It's annotation is shown below:<br><br>
![gl](https://cloud.githubusercontent.com/assets/16308037/24229340/ea8bad94-0fb3-11e7-9e1d-0a2217588ba4.jpg)
<br><br>
Finally, with solvepnp function in OpenCV, we can achieve real-time head pose estimation.
<br><br>
<b>Result:</b><br>
![head](https://cloud.githubusercontent.com/assets/16308037/24230147/79bf1c68-0fb8-11e7-859b-8482f9b559a5.gif)
