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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'common'))
import face_image
import face_preprocess
sys.path.append('..')
sys.path.append(os.path.join(os.path.dirname('__file__'), '..', 'RetinaFace'))
from RetinaFace.retinaface import RetinaFace



def do_flip(data):
  for idx in range(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

def get_model(ctx, image_size, model_str, layer):
  _vec = model_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  return model

class FaceModel:
  def __init__(self, args):
    self.args = args
    ctx = mx.gpu(args.gpu)
    _vec = args.image_size.split(',')
    assert len(_vec)==2
    image_size = (int(_vec[0]), int(_vec[1]))
    self.model = None
    self.ga_model = None
    if len(args.model)>0:
      self.model = get_model(ctx, image_size, args.model, 'fc1')
    if len(args.ga_model)>0:
      self.ga_model = get_model(ctx, image_size, args.ga_model, 'fc1')

    self.threshold = args.threshold
    self.det_minsize = 50
    self.det_threshold = args.det_threshold
    #self.det_factor = 0.9
    self.image_size = image_size
    retina_path = args.detector_path
    if args.det==0:
      detector = RetinaFace(retina_path,0, ctx_id=0)
    else:
      raise('Give detector path')
    self.detector = detector


  def get_input(self, face_img,scales=[1.0],flip=False):
    ret = self.detector.detect(face_img, self.det_threshold,scales=scales,do_flip=flip)
    if ret is None:
      return None,None,None,None
    bboxes, points = ret
    
    if bboxes.shape[0]==0:
      return None,None,None, None
    if bboxes is None:
        return None, None, None, None
    output=[]
    faces_=[]
    key_points_=[]
    bboxes_=[]
    if True:
        for face,point in zip(bboxes,points):
            bbox = face[0:4].astype(np.int)
            #return(bbox)
            # print (point.shape,point,bbox.shape,bbox)
            #point=point.reshape((2,5)).T
            to_add_face=face_img[bbox[1]:bbox[3],bbox[0]:bbox[2]]
            faces_.append(to_add_face)
            key_points_.append((point.astype(np.int),face[4]))
            bboxes_.append(bbox)
            

            nimg = face_preprocess.preprocess(face_img, bbox, point, image_size='112,112')
            #nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
            aligned = np.transpose(nimg, (2,0,1))
            output.append(aligned)
        return np.array(output),faces_,np.array(key_points_),np.array(bboxes_)
#     except:
#         import pdb; pdb.set_trace()
    
  def get_feature(self, aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    #print(input_blob.shape)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.model.forward(db, is_train=False)
    embedding = self.model.get_outputs()[0].asnumpy()
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding


  def get_batch_feature(self, aligned):
    input_blob = aligned
    #print(input_blob.shape)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.model.forward(db, is_train=False)
    embedding = self.model.get_outputs()[0].asnumpy()
    embedding = sklearn.preprocessing.normalize(embedding)
    return embedding

  def get_ga(self, aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.ga_model.forward(db, is_train=False)
    ret = self.ga_model.get_outputs()[0].asnumpy()
    g = ret[:,0:2].flatten()
    gender = np.argmax(g)
    a = ret[:,2:202].reshape( (100,2) )
    a = np.argmax(a, axis=1)
    age = int(sum(a))

    return gender, age

  def get_face_patch(self,img,bboxes,points):
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
        key_points_.append((point.astype(np.int),face[4]))
        bboxes_.append(bbox)
        #print(to_add_face.shape)
        
    return np.array(faces_),np.array(key_points_),np.array(bboxes_)
    
    