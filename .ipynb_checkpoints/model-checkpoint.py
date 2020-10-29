import  sys
sys.path.append('deploy/')
sys.path.append('RetinaFace')
from face_model_2 import FaceModel


class args:
    def __init__(self):
        self.image_size = '112,112'
        self.model = '/home/kakarot/Mask_detection/Mask_Detection/Face_detection/models/insightface/model-r100-ii/model,0'
        self.ga_model = ''
        self.gpu = 0
        self.det = 0
        self.flip =False
        self.threshold = 1.24
        self.detector_path = '/home/kakarot/Mask_detection/Mask_Detection/Face_detection/models/retinaface/R50'
        self.det_threshold = 0.7
        

Args = args()
Model  =  FaceModel(Args)
