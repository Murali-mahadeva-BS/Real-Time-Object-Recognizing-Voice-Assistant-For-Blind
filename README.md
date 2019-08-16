# Real-Time-Object-Recognizing-Voice-Assistant-For-Blind
With the use of Deep neural network in OpenCV objects are detected in real time and gives voice output. This will help blind people to move around independently.
This can be developed as standalone system by implementing the ssame on Raspberry pi.

-------------------------(WINDOWS)-------------------------
Requirements:

1. Python 3.x (32 bit or 64 bit)
   download link: https://www.python.org/downloads/

2. OpenCV (Computer vision library for python)
   Type in Command Prompt: "pip install opencv-python" 

3. gtts (Google Text To Speech)--converts text to speech
   Type in Command Prompt: "pip install gtts"

4. playsound - can play audio files in background without any media player
   Type in Command prompt: "pip install playsound"

-------------------------(LINUX)------------------------------
Requirements:

1. Python 3.x versions are preinstalled in popular linux distributions like Ubuntu & Fedora
   for installing type in Command line: "Sudo apt-get install python3"
 
2. Follow the below procedure to install OpenCV on linux
   https://docs.opencv.org/3.4.3/d7/d9f/tutorial_linux_install.html
   
   Follow the below link to install OpenCV on Raspberry Pi
   https://www.pyimagesearch.com/2016/04/18/install-guide-raspberry-pi-3-raspbian-jessie-opencv-3/

3. Installing gtts using 'pip'. pip comes pre-installed with python3 versions
   Command line: "Sudo pip install gtts " 

4. Installing playsound
  Command line: "Sudo pip install playsound"
  
-----------------------------------------------------------------------------------------------------------------

YOLO (You Only Look Once) is model used for object detection. After installing all the required software,
download configuration files from below link.

https://drive.google.com/open?id=11DUumWP2dR4uxIKhNvIQH_EysdVjWE9u

you need to import the below files to configure the model

1. 'coco.names' file contains all the classes(objects) that this model is trained to detect.
   To see the contents of this file, after importing the essentials, open this file and call a 'print' function in 'for loop'.
   
2. 'yolov3.weights' this file contains all the weights of the nodes of neural layers

3. 'yolov3.cfg' this file contains the biases of nodes or neurons

------------------------------CREATING NO OBJECT DETECTED MP3 FILE-------------------- 
1. Run the code no_obj.py

2. specify the path where you wish to save that mp3 file

3. Specify this 'no object detected' mp3 path in object recognition code

<Code written doesnt give visual output on the screen when you try to test it but gives voice output.
it depends on your system's capacity to how many number of frames it can process per second>
