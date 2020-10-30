# Mask Detection

<img src="1_huh2tZKYK3TwAulj_kUUqg.jpeg" width="300"/> 

## Details about this repo

Please read my article [here](https://medium.com/@harshshrm94/mask-detection-using-deep-learning-2958503d42b1), which has an explanation about how to detect mask given a video feed

## How To Use
* Clone this repo.
* Download models from [here](https://drive.google.com/drive/folders/1G6-UJuLdDPybbk-4Z3829kQNzN8bhbIj?usp=sharing)
* Make a directory "models/retinaface" inside Face_detection folder and extract **"retinaface-R50.zip"** in that folder and maake a directory "model" inside Mask_classification folder and put **"model_clean_data.pkl"** in that folder.
* Change the location to /Mask-detection/Face_detection by running:
```
cd Mask-detection/Face_detection
```
* Create the conda environment using the following command :
```
conda env create -n detection -f requirements.yml
```
* Change the pwd to Mask_classification by running :
```
cd Mask-detection/Mask_classification
```

* Create the conda environment using the following command :
```
conda env create -n classification -f requirements.yml
```
* Activate the Classification environment and run a server at port 3000 by running following commands
```
conda activate classification
cd Mask-detection/Mask_classification
flask run - -port 3000
```
* Run the following in a separate terminal. Keep the server running.
```
conda activate detection
cd loc/Mask-detection
```
* Then you can run the script in following way after activating the environment:
```
python infer.py --is_image True --in_path path/to/image --out_path path/to/save
```

You can also go through my blog [here](https://medium.com/@harshshrm94/mask-detection-using-deep-learning-part-ii-ab7a2cb6aaf1), where I have explained each step of implementation.
