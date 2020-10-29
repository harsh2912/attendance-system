from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
#import tensorflow as tf
import numpy as np
import mxnet as mx
import random
import cv2
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
# from mtcnn_detector import MtcnnDetector
sys.path.append(os.path.join(os.path.dirname('__file__'), '..', 'src', 'common'))
import face_image
import face_preprocess
sys.path.append('..')
sys.path.append(os.path.join(os.path.dirname('__file__'), '..', 'RetinaFace'))
from RetinaFace.retinaface import RetinaFace


class Detector:
    def __init__(self,model_path):
        self.model = RetinaFace(model_path,0, ctx_id=0)
        
    def get_face_patch(self,img):
        bboxes,points = self.model.detect(img, 0.7,scales=[1.0],do_flip=False)
        if isinstance(img,str):
            img=cv2.imread(img)
        faces_=[]
        key_points_=[]
        bboxes_=[]
        for face,point in zip(bboxes,points):
            #import pdb; pdb.set_trace()
            bbox = face[0:4].astype(np.int)
            to_add_face=img[bbox[1]:bbox[3],bbox[0]:bbox[2]]
            to_add_face=cv2.cvtColor(to_add_face, cv2.COLOR_BGR2RGB)
            faces_.append(to_add_face)
            key_points_.append((points.astype(np.int),face[4]))
            bboxes_.append(bbox)
            #print(to_add_face.shape)

        return faces_,np.array(key_points_),np.array(bboxes_)
    