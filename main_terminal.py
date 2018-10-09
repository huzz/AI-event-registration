import cv2
from utils.align_custom import AlignCustom
from utils.face_feature import FaceFeature
from utils.mtcnn_detect import MTCNNDetect
from utils.tf_graph import FaceRecGraph
import argparse
import sys
import json
import numpy as np
import time
import sys, os
from time import strftime, gmtime
import datetime
import csv
import time
count=0
vs = cv2.VideoCapture(0);

def main():
    #mode = args.mode
    #print("hereee")
    
    print("\n")
    print("PLEASE ENTER YOUR OPTION\n")
    print("Enter number 1 for REGISTRATION\n")
    print("Enter number 2 for DETECTION\n")
    mode = int(input(">>>"))
    if(mode == 1):
    	create_manual_data();
    elif mode == 2:
    	camera_recog()
    else:
        raise ValueError("Unimplemented mode")


def camera_recog():
    count=0
    recog_data=[('', 0)]
    print("[INFO] camera sensor warming up...")
    #vs = cv2.VideoCapture(0); #get input from webcam
    while True:
        _,frame = vs.read();
        rects, landmarks = face_detect.detect_face(frame,80);#min face size is set to 80x80
        aligns = []
        positions = []
        #print(rects)
        for (i, rect) in enumerate(rects):
            aligned_face, face_pos = aligner.align(160,frame,landmarks[i])
            if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
                aligns.append(aligned_face)
                positions.append(face_pos)
            else: 
                print("Align face failed") #log 
            prev_recog = recog_data[0][0]       
            if(len(aligns) == 1 and face_pos=='Center') :
                # print(len(aligns))
                features_arr = extract_feature.get_features(aligns)
                recog_data = findPeople(features_arr,positions);
                cv2.rectangle(frame,(rect[0],rect[1]),(rect[0] + rect[2],rect[1]+rect[3]),(255,0,0))
                cv2.putText(frame,recog_data[0][0]+" - "+str(recog_data[0][1])+"%",(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
               
        cv2.imshow("Frame",frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
        	cv2.destroyAllWindows()
        	main()

def findPeople(features_arr, positions, thres = 0.6, percent_thres = 60):

    f = open('./face_dataset.txt','r')
    data_set = json.loads(f.read());
    returnRes = [];
    for (i,features_128D) in enumerate(features_arr):
        result = "Unknown";
        smallest = sys.maxsize
        for person in data_set.keys():
            person_data = data_set[person][positions[0]];
            for data in person_data:
                distance = np.sqrt(np.sum(np.square(data-features_128D)))
                if(distance < smallest):
                    smallest = distance;
                    result = person;
        percentage =  min(100, 100 * thres / smallest)
        if percentage <= percent_thres :
            result = "Unknown"
        returnRes.append((result,percentage))

    return returnRes


def create_manual_data():
    #vs = cv2.VideoCapture(0); #get input from webcam
    
    recval=0
    ans = ' '
    click = True
    print(" Please enter your Full Name :)")
    full_name = input()
    
    while (True):
    	f = open('./face_dataset.txt','r');
    	data_set = json.loads(f.read());
    	t=open('./time_dataset.txt','r');
    	time_data=json.loads(t.read());

    	person_imgs = {"Center": []};
    	person_features = {"Center": []};
    	person_time={"Hour":[],"Min":[],"Sec":[]};

    	while (click):
    		_,frame = vs.read();
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
    			print(final_name)
    			data_set[final_name] = person_features;
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
    			f.write(json.dumps(data_set))

    			print("Welcome ",final_name)
    			print(" ")

    			while True:
	    			img = np.zeros((800,1400,3), np.uint8)
	    			font = cv2.FONT_HERSHEY_COMPLEX
	    			cv2.putText(img,"Welcome to IoTIoT :)",(50,300), font, 3,(47, 160, 181),6,cv2.LINE_AA)
	    			cv2.putText(img,final_name,(50,500), font, 3,(226, 185, 61),6,cv2.LINE_AA)
	    			cv2.imshow("Thank you for your Registration", img)
	    			key = cv2.waitKey(1) & 0xFF
	    			if key == ord("q"):
	    				cv2.destroyAllWindows()
	    				break
    		
    			recval=0
    			t.close()
    			global count
    			count=count+1
    			click = False
    			cv2.destroyAllWindows()
    			print("press 'y' to continue ?")
    			ans = input()
    			break;
    		break
    	if(ans == 'y'):
    		main()




if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--mode", type=str, help="Run camera recognition", default="camera")
    #args = parser.parse_args(sys.argv[1:]);
    FRGraph = FaceRecGraph();
    aligner = AlignCustom();
    extract_feature = FaceFeature(FRGraph)
    face_detect = MTCNNDetect(FRGraph, scale_factor=2); #scale_factor, rescales image for faster detection
    main();
