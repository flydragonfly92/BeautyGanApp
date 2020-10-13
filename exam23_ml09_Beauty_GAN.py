#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


# In[2]:


detector = dlib.get_frontal_face_detector()  # dlib에서 얼굴을 찾아주는 기능을 수행하는 함수
sp = dlib.shape_predictor('../models/shape_predictor_5_face_landmarks.dat')
# 해당 파일은 dilb에서 얼굴을 찾아주는 기능을 수행하는 모델을 담고 있다.
# 5개의 랜드마크로 얼굴을 찾아줌


# In[3]:


img = dlib.load_rgb_image('../imgs/12.jpg')  # 이미지 불러오기
plt.figure(figsize=(16,10))  # 화면에 출력하기
plt.imshow(img)
plt.show()


# In[4]:


img_result = img.copy()
dets = detector(img)  # dlib의 detector 함수는 얼굴을 찾아주는 기능을 수행한다.
if len(dets) == 0:
    print('cannot find faces!')
else:
    fig, ax = plt.subplots(1,figsize=(16,10))
    for det in dets:
        x, y, w, h = det.left(), det.top(), det.width(), det.height()
        rect = patches.Rectangle((x,y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # 이미지는 원점의 위치가 좌측 상단을 기준으로 한다.
        # linewidth=2  => 선의 넓이를 나타냄
        # edgecolor='r' => 테두리 선의 색깔을 나타냄
        # facecolor='none' => 얼굴색깔을 없음을 나타냄 (cf) facecolor='g' 로 하면 사각형이 초록색으로 된다.
        # add_patch는 그림을 화면에 띄우는 함수
    ax.imshow(img_result)
    plt.show()


# In[5]:


fig, ax = plt.subplots(1, figsize=(16,10))
objs = dlib.full_object_detections()
for detection in dets:
    s = sp(img, detection)
    objs.append(s)
    # parts는 점을 그리는 기능을 수행하는 함수
    for point in s.parts():
        circle = patches.Circle((point.x, point.y), radius=3, edgecolor='r', facecolor='r')
        ax.add_patch(circle)
ax.imshow(img_result)


# In[6]:


faces = dlib.get_face_chips(img, objs, size=256, padding=0.3)
fig, axes = plt.subplots(1, len(faces)+1, figsize=(20,16)) 
# subplots(row 개수, 원본이미지 + 원본이미지에 있는 각 얼굴 개수, 얼굴만 추출한 이미지의 사이즈)
axes[0].imshow(img)
for i, face in enumerate(faces):
    axes[i+1].imshow(face)


# In[7]:


def align_faces(img):
    dets = detector(img, 1)  # 얼굴 찾기, dets는 얼굴에 대한 영역(범위 및 좌표)관련 정보를 가지고 있다.
    objs = dlib.full_object_detections()  # 이미지를 구분하기 위해 이미지에 대한 정보를 갖고 있는 객체. 그러나 처음에는 공란이다.
    for detection in dets:
        s = sp(img, detection)  # 얼굴에 있는 랜드마크(점) 정보를 나타낸다.
        objs.append(s)  # 얼굴마다 랜드마크를 붙여 objs에 저장된다.
    faces = dlib.get_face_chips(img, objs, size=256, padding=0.35)  # 랜드마크된 얼굴 범위를 추출해서 얼굴 이미지로 나타낸다.
    # padding은 추출한 얼굴 사진에 상하좌우 여분을 추가해서 추출한다. => 얼굴 외에 얼굴 근처 사진도 같이 추출되는 효과 
    return faces
test_img = dlib.load_rgb_image('../imgs/03.jpg')  # 이미지 파일 불러옴
test_faces = align_faces(test_img)  # 위에 있는 함수를 실행
fig, axes = plt.subplots(1, len(test_faces)+1, figsize=(20,16))  # subplot을 만든다.
axes[0].imshow(test_img)  # axes[0]는 원본 이미지를 나타냄. 원본 이미지를 출력한다.
for i, face in enumerate(test_faces):  # 얼굴 이미지들을 subplot에 출력한다.
    axes[i+1].imshow(face)


# In[8]:


sess = tf.Session()
# keras는 .model로 모델을 실행하는 반면 tensorflow는 .run으로 모델을 실행함
sess.run(tf.global_variables_initializer())
# tensorflow는 .import로 모델을 불러온다
saver = tf.train.import_meta_graph('../models/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('../models'))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')


# In[9]:


def preprocess(img):
    return(img/255. -0.5)*2

def deprocess(img):
    return(img+1) / 2


# In[21]:


# 폴더에 있는 이미지를 불러옴 (img1은 no_makeup, img2는 makeup)
img1 = dlib.load_rgb_image('../imgs/12.jpg')
img1_faces = align_faces(img1)

img2 = dlib.load_rgb_image('../imgs/makeup/vFG56.png')
img2_faces = align_faces(img2)

fig, axes = plt.subplots(1, 2, figsize=(16,10))
axes[0].imshow(img1_faces[0])
axes[1].imshow(img2_faces[0])
plt.show()


# In[22]:


src_img = img1_faces[0]
ref_img = img2_faces[0]

X_img = preprocess(src_img)
X_img = np.expand_dims(X_img, axis=0)  # 모델에 입력하기 위해 shape을 맞춰줌

Y_img = preprocess(ref_img)
Y_img = np.expand_dims(Y_img, axis=0)

# Model에 input값을 대입 -> dict 형태로 입력
# keras는 model.fit()으로 입력하는 반면 tensorflow는 session.run()으로 입력한다.
output = sess.run(Xs, feed_dict={X:X_img, Y:Y_img})
output_img = deprocess(output[0])

fig, axes = plt.subplots(1, 3, figsize=(20,10))
axes[0].set_title('Source')
axes[0].imshow(src_img)
axes[1].set_title('Reference')
axes[1].imshow(ref_img)
axes[2].set_title('Result')
axes[2].imshow(output_img)
plt.show()


# In[ ]:




