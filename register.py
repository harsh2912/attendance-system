import os
import numpy as np
from functools import partial
import cv2
from concurrent.futures import ProcessPoolExecutor as ppe
import  sys
import redis
import struct
from model import Model
import argparse

sys.path.append('RetinaFace/rcnn')


class Registration:
    def __init__(self,redis_conn:redis.Redis,model:Model):
        self.model = Model
        self.redis = redis_conn
        
    def register(self,path):
        try:
            
            vectors = self.redis.get('vectors')
            
            if vectors is not None: 
                h,w = struct.unpack('>II',vectors[:8])           #getting the shape of the vectors as it is stored as bytes
                array = np.frombuffer(vectors[8:])
                array = array.reshape(h,w)
                index = h
            else:
                array = np.zeros((1,512))
                index = 0
                
            images, names = self.load_images(self,path)
            vectors = self.get_vectors(images,array) if index != 0 else self.get_vectors(images,array)[1:] #removing zero vector if registering first time
            
            with self.redis.pipeline() as pipe:     #updating mapping of indices of vector matrix to names
                for j,name in zip(range(index,len(names)+index),names):
                    pipe.hset('mapping',j,name)
                pipe.execute()
                
            shape = struct.pack('>II',vectors.shape[0],vectors.shape[1])

            vectors = shape + vectors.tobytes()

            self.redis.set('vectors',vectors)        #updating vectors
            self.redis.bgsave()
            
        except(redis.ConnectionError):
            print('The Redis Server is not UP!')
            return
        

    
    @staticmethod
    def load_images(self,path):
        all_images = []
        student_names = os.listdir(path)
        student_names.remove('.ipynb_checkpoints')
        for student in student_names:
            student_image = []
            image_paths = [f'{path}/{student}/{i}' for i in os.listdir(f'{path}/{student}/') if '.ipynb' not in i]
            with ppe() as e:
                for output in e.map(cv2.imread,image_paths):  #multiprocessing to load all the images from a folder
                    student_image.append(output)
            all_images.append(student_image)
        return all_images, student_names

    
    def get_vector(self,images):
        all_alligned_faces = []
        for img in images:
            #getting the alligned faces using retinaface to detect the face first
            alligned_faces,faces,kps,boxes = self.model.get_input(img[:,:,::-1])
            all_alligned_faces.append(alligned_faces)

        #stacking the faces as batch to process in a batch for vector generation
        faces = np.vstack(all_alligned_faces)
        vectors = self.model.get_batch_feature(faces)
        return vectors.mean(axis=0) #Returning the mean of all the vectors generated
    
    def get_vectors(self,images:list,array:np.array):
        
        for image_set in images:
            vector = self.get_vector(image_set)
            array = np.vstack((array,vector))
        return array
            
            


parser = argparse.ArgumentParser()
parser.add_argument('-p','--path',help='path to faces',type=str)
parser.add_argument('-db','--database',help='redis database',type=str)

if __name__=='__main__':
    args = parser.parse_args()
    db = args.database
    path = args.path
    r = redis.Redis(db=db)
    
    Register = Registration(r,Model)
    print('Registering.........')
    Register.register(path)
    print('Registeration Done')
    