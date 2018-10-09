# AI Registration System
This repo contains an application for automated face registration using AI. 
It is a smart AI based, user-friendly face registration application to reduce the hassle of registration desks


MTCNN model is used for face detection and Facenet model is used to extract embeddings of the reistered face. Hence Face is recognised. The embeddings of the person are stored in face_dataset.txt and the time and date of their registration is stored in time_dataset. 
 

#### How to run the code : ####

1. Install the dependencies

```bash
  pip3 install -r requirements.txt
```

2. Run the code :

For GUI output
 ```bash
 python3 main_gui.py
 ```
For command line interface
 ```bash
 python3 main_terminal.py 
 ```

3. Screenshots

Register:
![alt text](https://github.com/huzz/AI-event-registration/blob/master/register.png)

Detect:
![alt text](https://github.com/huzz/AI-event-registration/blob/master/detect.png)
