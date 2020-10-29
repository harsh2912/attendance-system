from flask import Flask,jsonify,request,send_from_directory
from flask_cors import CORS
import base64
import os
from werkzeug.utils import secure_filename
app = Flask(__name__)
CORS(app)
app.debug = True
from flask import Flask,jsonify
from face_detection import Model


retina_path = './models/retinaface/R50'
req_add = 'http://127.0.0.1:3000/'
save_path = 'to_serve_files/output.avi'
model = Model(retina_path,req_add)
all_image_formats = ['.jpeg',
 '.jfif',
 '.exif',
 '.tiff',
 '.bmp',
 '.png',
 '.ppm',
 '.pgm',
 '.pbm',
 '.pnm',
 '.webp',
 '.hdr',
 '.heif',
 '.bat']

all_video_formats = ['.webm',
 '.mpg',
 '.mp2',
 '.mpeg',
 '.mpe',
 '.mpv',
 '.ogg',
 '.mp4',
 '.m4p',
 '.m4v',
 '.avi',
 '.wmv',
 '.mov',
 '.qt',
 '.flv',
 '.swf',
 '.avchd'
   '.gif']
main_ext = '.avi'

@app.route("/",methods=["POST"])
def index():
    global main_ext 
    #print("inside route")
    if request.method == "POST":
        #print("inside route 1")
        
#         flag = request.json()['']
        content=request.files['file']
        fname,ext=os.path.splitext(content.filename)
        #print("extension",ext)
        #main_ext = ext
#         print(ext)
        content.save("current_file"+ext)
        video_path = 'current_file'+ext
        if ext.lower() in all_video_formats:
            type_ = 'video'
            print('video')
        elif ext.lower() in all_image_formats:
            type_ = 'image'
            print('image')
        model.generate_output(video_path,save_path,type_)

        print('sent from main')
        return jsonify({'success': True})

@app.route("/file")
def get_file():
    global main_ext
    print(main_ext)
    return send_from_directory("to_serve_files", f"output"+main_ext, as_attachment=True)



if __name__ == "__main__":
    app.run(debug=True)