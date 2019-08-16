import cv2
import numpy as np
import time,os
from gtts import gTTS
from playsound import playsound

    
#load YOLO
net=cv2.dnn.readNet("C:\\Users\\vcs\\Desktop\\project\\asssistant for blind\\yolov3.weights","C:\\Users\\vcs\\Desktop\\project\\asssistant for blind\\yolov3.cfg")

with open("C:\\Users\\vcs\\Desktop\\project\\asssistant for blind\\coco.names",'r') as f:
    classes=[line.strip() for line in f]

layer_names=net.getLayerNames()

output_layers=[layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

cap=cv2.VideoCapture(0)

start_time=time.time()
frame_no=0

inc=0
while True:
    _,frame=cap.read()
    frame_no+=1
    class_ids=[]
    confidences=[]
    detect_obj=0
    
    height,width = frame.shape[:2]

    blob=cv2.dnn.blobFromImage(frame,0.00392,(416,416),(0,0,0),True,crop=False)
                  
    net.setInput(blob)

    outputs = net.forward(output_layers)
   
    for out in outputs:
        for i in out:
            scores = i[5:]
            class_id = np.argmax(scores)
            confidence=scores[class_id]
            if confidence>0.6:
                #object detected
                class_ids.append(class_id)
                confidences.append(float(confidence))

    for i in range(len(class_ids)):
        conf=confidences[i]
        label = classes[class_ids[i]]
        print(label, conf*100)
        
        voice=str(label)+"in front of you"
        
    file_path='C:/Users/vcs/Desktop/project/asssistant for blind/temp/sound'+str(inc)+'.mp3'
    inc+=1
    sound=gTTS(text=voice,lang='en')
    sound.save(file_path)
    if class_ids:
        playsound(file_path)
    else:
        playsound('C:/Users/vcs/Desktop/project/asssistant for blind/temp/no_obj.mp3')
    os.remove(file_path)
    end_time=time.time()
    elapsed=end_time-start_time
    print(elapsed/frame_no)
cap.release()
cv2.destroyAllWindows()

