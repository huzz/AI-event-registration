from tkinter.ttk import *
import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
from utils.align_custom import AlignCustom
from utils.face_feature import FaceFeature
from utils.mtcnn_detect import MTCNNDetect
from utils.tf_graph import FaceRecGraph
import argparse
import sys
import json
from time import strftime, gmtime
import datetime
import csv
import time

global last_frame                                 
last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
global cap
cap = cv2.VideoCapture(0)
global det_peeps
det_peeps=0
global count
count=0
global member
def retrieve_input(textBox,inp):
    global member
    member=textBox.get("1.0","end-1c")
    inp.destroy()
    inp.quit()


def register():
	# count=0
	recval=0
	click = True

	inp=Tk()
	info=Label(inp, height=5, width=30,text="Enter your your full name",font=("Helvetica bold", 20))
	textBox=Text(inp, height=1, width=20,font=("Helvetica bold", 25))
	info.pack()
	textBox.pack()
	buttonCommit=tk.Button(inp, height=4, width=15, text="Submit",command=lambda:retrieve_input(textBox,inp),activebackground="black", activeforeground="cyan", bd=4, bg="#123d63", fg="gold", font=("Helvetica bold", 15))
	#command=lambda: retrieve_input() >>> just means do this when i press the button
	
	buttonCommit.pack()
	mainloop()
	global member
	print(member)



	full_name = member
	
	while (click):
		f = open('./face_dataset.txt','r');
		face_data = json.loads(f.read());
		t=open('./time_dataset.txt','r');
		time_data=json.loads(t.read());

		person_imgs = {"Center": []};
		person_features = {"Center": []};
		person_time={"Hour":[],"Min":[],"Sec":[]};

		while True:
			_,frame = cap.read();
			rects, landmarks = face_detect.detect_face(frame, 80);  # min face size is set to 80x80
			for (i, rect) in enumerate(rects):
				cv2.rectangle(frame,(rect[0],rect[1]),(rect[0] + rect[2],rect[1]+rect[3]),(255,0,0), 6) #draw bounding box for the face
				aligned_frame, pos = aligner.align(160,frame,landmarks[i]);
				if len(aligned_frame) == 160 and len(aligned_frame[0]) == 160:
					if(pos=="Center"):
						person_imgs[pos].append(aligned_frame)
			cv2.imshow("Register Face",frame)
			key = cv2.waitKey(1) & 0xFF

			if rects == []:
				recval=0
				break
			else:
				recval=recval+1

			if (recval>=7):
				for pos in person_imgs:
					person_features[pos] = [np.mean(extract_feature.get_features(person_imgs[pos]),axis=0).tolist()]

				final_name=full_name
				#print(final_name)
				face_data[final_name] = person_features;
				currh=datetime.datetime.now().strftime('%H')
				currm=datetime.datetime.now().strftime('%M')
				currs=datetime.datetime.now().strftime('%S')
				person_time["Hour"]=currh
				person_time["Min"]=currm
				person_time["Sec"]=currs
				time_data[final_name]=person_time;
				t = open('./time_dataset.txt', 'w');
				t.write(json.dumps(time_data))
				f = open('./face_dataset.txt', 'w');
				f.write(json.dumps(face_data))

				print("Welcome ",final_name)
				print(" ")

				while(True):

					img = np.zeros((800,1400,3), np.uint8)
					#cv2.line(img,(0,0),(511,511),(255,0,0),5)
					font = cv2.FONT_HERSHEY_COMPLEX
					cv2.putText(img,"Welcome to the event:)",(50,300), font, 3,(47, 160, 181),6,cv2.LINE_AA)
					cv2.putText(img,final_name,(50,500), font, 3,(226, 185, 61),6,cv2.LINE_AA)
					cv2.imshow("Press 'q' to close window", img)
					key = cv2.waitKey(1) & 0xFF
					if key == ord("q"):
						cv2.destroyAllWindows()
						break

				t.close()
				recval=0
				global count
				count=count+1
				click = False
				cv2.destroyAllWindows()
				break;

def detect():
	print("Detecting People")
	det_peeps=0
	detect=0
	recog_data=[('', 0)]
	
	while(True):
		ret,frame=cap.read()
		rects, landmarks = face_detect.detect_face(frame,80);#min face size is set to 80x80
		aligns = []
		positions = []
		for (i, rect) in enumerate(rects):
			aligned_face, face_pos = aligner.align(160,frame,landmarks[i])
			if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
				aligns.append(aligned_face)
				positions.append(face_pos)
			else: 
				print("Align face failed")

			if(len(aligns) == 1 and face_pos=='Center') :
				features_arr = extract_feature.get_features(aligns)
				recog_data = findPeople(features_arr,positions);
				per = recog_data[0][0]

				cv2.rectangle(frame,(rect[0],rect[1]),(rect[0] + rect[2],rect[1]+rect[3]),(0,255,0), 4)
				cv2.putText(frame,str(recog_data[0][0])+" - " + str(recog_data[0][1])+"%",(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)


		cv2.imshow('Detected People. Press q to exit.',frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			cv2.destroyAllWindows()
			break
	

def findPeople(features_arr, positions, thres = 0.99, percent_thres = 99):
	f = open('./face_dataset.txt','r')
	face_data = json.loads(f.read());
	returnRes = [];
	for (i,features_128D) in enumerate(features_arr):
		result = "Unknown";
		smallest = sys.maxsize
		for person in face_data.keys():
			person_data = face_data[person][positions[0]];
			for data in person_data:
				distance = np.sqrt(np.sum(np.square(data-features_128D)))
				if(distance < smallest):
					smallest = distance;
					# print("Smallest",smallest)
					result = person;
		percentage =  min(100, 100 * thres / smallest)

		if percentage <= percent_thres :
			result = "Unknown"
		returnRes.append((result,percentage))
	return returnRes

def quit():
	root.quit()

if __name__ == '__main__':
	FRGraph = FaceRecGraph();
	aligner = AlignCustom();
	extract_feature = FaceFeature(FRGraph)
	face_detect = MTCNNDetect(FRGraph, scale_factor=2); 
	y=2
	root=tk.Tk()                                 
	lmain = tk.Label(master=root)
	lmain.grid(row=0,column=0, rowspan=10, padx=10, pady=10)
	a = Button(master=root, text="REGISTER",height=6,width=22,command=register,activebackground="black", activeforeground="cyan", bd=4, bg="#123d63", fg="gold", font=("Helvetica bold", 15))
	a.grid(row=1,column=0, padx=5, pady=5)
	b = Button(master=root, text="DETECT",height=6,width=22,command=detect, activebackground="black", activeforeground="cyan", bd=4, bg="#123d63", fg="gold", font=("Helvetica bold", 15))
	b.grid(row=2,column=0, padx=5, pady=5)
	c = Button(master=root, text="EXIT",height=6,width=22,command=quit, activebackground="black", activeforeground="cyan", bd=4, bg="#123d63", fg="gold", font=("Helvetica bold", 15))
	c.grid(row=3,column=0, padx=5, pady=5)
	root.title("Event Registration")   
	root.mainloop()                                  
cap.release()