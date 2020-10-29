import cv2
import redis
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys
import struct
from model import Model
import time


parser = argparse.ArgumentParser()
parser.add_argument('-in','--in_path',help='Video path',type=str)
parser.add_argument('-out','--out_path',help='output video path',type=str)
parser.add_argument('-db','--database',help='redis database',type=str)



if __name__=='__main__':
#     start = time.time()
    args = parser.parse_args()
    in_path = args.in_path
    out_path = args.out_path
    redis_db = args.database
    
    
    cap = cv2.VideoCapture(in_path)
    ret,fr = cap.read()
    r = redis.Redis(db=redis_db)
    threshold = 0.23
    
    try:
    
        stored_vectors = r.get('vectors')
        mapping = r.hgetall('mapping')
        
    except(redis.ConnectionError):
        
        raise("Redis Server is not UP!")
        
    
    if stored_vectors is None:
        raise('Please Register First!')
#     print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    h,w = struct.unpack('>II',stored_vectors[:8])           #getting the shape of the vectors as it is stored as bytes
    stored_vectors = np.frombuffer(stored_vectors[8:])
    stored_vectors = stored_vectors.reshape(h,w)
    
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    height,width = fr.shape[1],fr.shape[0]
    writer = cv2.VideoWriter(f'{out_path}/output.avi', fourcc, frame_rate, (height,width))
    
    while ret:
        
        alligned_faces,faces,kps,boxes = Model.get_input(fr[:,:,::-1])
        
        if boxes is None:
            writer.write(cv2.resize(fr,(height,width)))
            ret,fr = cap.read()
            continue
        
        generated_vectors = Model.get_batch_feature(alligned_faces)
        
        sim  = cosine_similarity(stored_vectors,generated_vectors)
        
        max_sim_indices = sim.argmax(axis=0)
        
        matched_faces = np.where(sim[max_sim_indices,range(len(generated_vectors))]>threshold)[0]
        
        left_faces = [i for i in range(len(generated_vectors)) if i not in matched_faces]
        
        if len(matched_faces) != 0:
        
            matched_stored_faces = max_sim_indices[matched_faces]

            for stored_idx,gen_idx in zip(matched_stored_faces, matched_faces):
                box = boxes[gen_idx]
                
                cv2.rectangle(fr, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
                name = mapping[f'{stored_idx}'.encode()].decode()
                
                cv2.putText(
                        fr,str(name), 
                        (box[0],box[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1,
                        (0,255,0),
                        2
                           )
        for idx in left_faces:
            box = boxes[idx]
            cv2.rectangle(fr, (box[0], box[1]), (box[2], box[3]), (0,0,255), 2)
            
        fr = cv2.resize(fr,(height,width))
        
        writer.write(fr)
        ret,fr = cap.read()
        
    print('Done')
#         frame_id+=1
#     print(time.time()-start)
            
            
        
        