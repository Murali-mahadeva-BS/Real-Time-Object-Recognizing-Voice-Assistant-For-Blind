#import all the essentials
import cv2
import numpy as np
import time,os
from gtts import gTTS
from playsound import playsound

    
#load YOLO
net=cv2.dnn.readNet("yolov3.weights","yolov3.cfg")# can add path also

#loading coco.names file, which contains names of objects it can detect
with open("coco.names",'r') as f:               
    classes=[line.strip() for line in f]

layer_names=net.getLayerNames()

output_layers=[layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

cap=cv2.VideoCapture(0) #you can give 1 or 2 for extra connected webcams  
frame_no=0
inc=0

#Looping creates N_images to look like video
while True:
    start_time=time.time()  #starting time counting to measure frames processing speed 
    _,frame=cap.read()  #reading from webcam
    frame_no+=1
    class_ids=[]
    confidences=[]
    detect_obj=0
    
    height,width = frame.shape[:2]  #gives dimensions of current frame

    blob=cv2.dnn.blobFromImage(frame,0.00392,(416,416),(0,0,0),True,crop=False) #detects blob(group of identicals) within the frame
                  
    net.setInput(blob)

    outputs = net.forward(output_layers)
   
    for out in outputs:
        for i in out:                       # the 'i' in 'out' is a list of 85 numbers. Real sccores are from index 5 to 85 whose value are between 0 and 1
            scores = i[5:]                  # cuts 85 to 80 required numbers
            class_id = np.argmax(scores)    #gives index of max valued number in above list of 80 numbers
            confidence=scores[class_id]     #gives highest score in list of 80 values which are between 0 and 1
            if confidence>0.6:
                #object detected
                class_ids.append(class_id)  #all objects and their respective confidences of all blobs are stored as a list
                confidences.append(float(confidence))

    for i in range(len(class_ids)):
        conf=confidences[i]
        label = classes[class_ids[i]]      #'label' variable holds name of object detected
        print(label, conf*100)             #prints object detected with how confident it is in its predition
        
        voice=str(label)+"in front of you" #string being passed to convert to voice with gtts
        
    file_path='voice{}.mp3'.format(inc)    #u can specify path to temporarily store text to voice conversion
    inc+=1
    sound=gTTS(text=voice,lang='en')       #text to voice conversion with gtts
    sound.save(file_path)                  #voice file saving in specified path
    if class_ids:   #if any object is detected it says the name else says no 'no object detected'
        playsound(file_path)
    else:
        playsound('no_obj.mp3') #create an mp3 file saying 'no object detected' refer README
    os.remove(file_path)    #removes the voice file saved
    end_time=time.time()    #stoping time counter after all processing is done  
    elapsed=end_time-start_time
    print(1/elapsed) #gives the number of frames processed per second
cap.release()
cv2.destroyAllWindows()

