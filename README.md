# Attendance System Using Face Recognition

<img src="friends.gif" width="500"/> 

## Details about this repo

Please read my article [here](https://medium.com/p/f49cd9bec02c/edit), which has an explanation about how an attendance system using Face Recognition, given a video feed works and read this article [here](https://medium.com/p/7e1858b66cd3/edit) to know how to implement it using this repository.

## How To Use
* Clone this repo.
* Download retinaface and insightface models from [here](https://drive.google.com/drive/folders/1G6-UJuLdDPybbk-4Z3829kQNzN8bhbIj?usp=sharing)
* Make a directory **"models"** inside **Face_detection** folder and extract **"retinaface-R50.zip"** in a folder inside **"models"** with name **"retinaface"** and extract **"insightface.zip"** in the **"models"** folder only. So now you will have two directories inside **models**, namely **insightface** and **retinaface**.
* In the root folder there is a file **environment.yml** which we will use to create a conda environment byt executing :
```
conda env create -n face_recog -f environment.yml
```
* Start the redis server in which we will store the vectors of registered faces by running :
```
redis-server
```
* Activate the conda environment created in previous step by running :
```
conda activate face_recog
```
* Register all the different person using **register.py**. To know about how to structure the data for registering please go through my [implementation](https://medium.com/p/7e1858b66cd3/edit) article
```
python register.py -p path/to/folder/ -db 0
```
* Run the following command to infer on a video feed after registering all the people :
```
python infer.py -in path/to/video -out to/save/path -db 0
```
