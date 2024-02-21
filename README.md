#Authors

@ahmetcoko

@HuseyınKanat

@yusufbayindir


#Dependencies

PyQt5: Used for creating the graphical user interface.

OpenCV (cv2): Utilized for image and video processing tasks.

NumPy: Employed for numerical operations and array manipulation.

Ultralytics YOLO: Utilized for object detection tasks.


#Overview

Smart Crowd Detection is a project developed as part of a CENG483(Behavioral Robotics) course term project. The aim of this project is a simultaneous and total bird's eye heat map visualization that provides an intuitive visualization of high-traffic areas and tracks the movements of individuals with the help of YOLOv8  in the space using the homography transformation method over a single camera orientation by assigning different color intensities to different areas depending on the density of pedestrian traffic, the heat map can provide valuable information on space usage and passenger behavior by highlighting areas with high activity or traffic congestion.

Perpective transformation on a frame
![image](https://github.com/ahmetcoko/Smart-Crowd-Detection/assets/158578355/20de6b62-1570-4a64-bf0d-a5b978e08436)



We utilized PyQt5 to create a simple user interface for the program.
![image](https://github.com/ahmetcoko/Smart-Crowd-Detection/assets/158578355/62dfc960-556b-43f2-a150-d05c71878940)


When you run the program, in order to perform the perpective transformation process more effectively, you need to select the corners of the ground as upper left, upper right, lower right and lower left, respectively.

![image](https://github.com/ahmetcoko/Smart-Crowd-Detection/assets/158578355/797e1f66-5b4f-4982-a127-6735c7383142)

![image](https://github.com/ahmetcoko/Smart-Crowd-Detection/assets/158578355/242e07fb-e801-43cf-b7ac-86219d2a2b99)


After clicking the "Start Analyze" button, the top-left window of the three windows that will open only displays the selected background (the heatmap background to be created), the top-right window displays the normal state of the selected video, and the bottom-left window displays the instantaneous heatmap generated while the video is streaming. Upon completion of the video or clicking the "Finish Process" button, it displays the total heatmap generated.

![image](https://github.com/ahmetcoko/Smart-Crowd-Detection/assets/158578355/be21d3ab-4fd9-4828-8fa2-8095b0eb6db7)

![image](https://github.com/ahmetcoko/Smart-Crowd-Detection/assets/158578355/766e86b6-06c9-416b-a08b-35a7a32d3d37)

![image](https://github.com/ahmetcoko/Smart-Crowd-Detection/assets/158578355/e6f6f314-a05e-4df4-aca2-1443a4fd04a5)

![image](https://github.com/ahmetcoko/Smart-Crowd-Detection/assets/158578355/bd29fad4-363a-4ffe-9410-4a0f6073a460)


If you believe that the video stream is progressing slowly, you can adjust the DETECTION_FREQUENCY variable to determine how often the process should be repeated, in terms of frames.
Additionally, you can use the program simultaneously via your mobile phone's camera.



