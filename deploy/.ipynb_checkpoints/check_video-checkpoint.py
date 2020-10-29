import face_model_2
import argparse
import cv2
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def cosine(f1, f2):
    return cosine_similarity(f1.reshape(1, -1), f2.reshape(1, -1))[0][0]


parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='/home/kakarot/Face_recognition_new/insightface/models/insightface/model-r100-ii/model,0', help='model_path')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--detector_path', default='../models/retinaface/R50', help='path of detector')
parser.add_argument('--det_threshold', default=0.8, type=float, help='detector threshold')
args = parser.parse_args()

model = face_model_2.FaceModel(args)
img = cv2.imread('../Videos/zLeVJhN-friends-tv-show-wallpapers.jpg')
aligned_faces,bbox,points = model.get_input(img)
faces , keypoints , boxes = model.get_face_patch(img,bbox,points)

saved_faces = os.listdir('../Output')

saved_faces = [i for i in saved_faces if not '.' in i]
for face,im,box in zip(faces,aligned_faces,boxes):
    sim = []
    f1 = model.get_feature(im)
    for saved_face in saved_faces:
        
        feature = np.load(f'../Output/{saved_face}/feature.npy')
        sim.append(cosine(f1,feature))
    print(sim)
    saved_name = saved_faces[np.argmax(sim)]
    plt.imsave(f'../Output/{saved_name}/face2.jpg',face)
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
plt.imsave('../Output/result2.jpg',img[...,::-1])

cap = cv2.VideoCapture(video_path)
ret , frame = cap.read()
while ret:
    saved_faces = os.listdir('../Video_faces')
    saved_faces = [i for i in saved_faces if not '.' in i]
    if not len(saved_faces) == 0:
        saved_features = [np.load(f'../Video_faces/{i}/feature.npy' for i in saved_faces)]
    aligned_faces,bbox,points = model.get_input(frame)
    faces , keypoints , boxes = model.get_face_patch(frame,bbox,points)
    
    for face,im,box in zip(faces,aligned_faces,boxes):
        sim = []
        feature = model.get_feature(im)
        if saved_features:
            for face_feature in saved_features:
                sim.append(cosine(feature,face_feature))
            if max(sim) > 0.6:
                name = saved_faces[np.argmax(sim)]
            else:
                name = str(np.random.randint(1,10000))
                while name in saved_faces:
                    name = str(np.random.randint(1,10000))
        else:
            name = str(np.random.randint(1,10000))
            while name in saved_faces:
                name = str(np.random.randint(1,10000))
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
#print(f1[0:10])
# #gender, age = model.get_ga(img)
# print(gender)
# print(age)
# sys.exit(0)
# img = cv2.imread('/raid5data/dplearn/megaface/facescrubr/112x112/Tom_Hanks/Tom_Hanks_54733.png')
# f2 = model.get_feature(img)
# dist = np.sum(np.square(f1-f2))
# print(dist)
# sim = np.dot(f1, f2.T)
# print(sim)
# #diff = np.subtract(source_feature, target_feature)
# #dist = np.sum(np.square(diff),1)
