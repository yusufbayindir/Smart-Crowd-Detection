from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt,QSize
from PyQt5.QtCore import pyqtSignal
import sys
import threading
import cv2
import numpy as np
from ultralytics import YOLO
from distutils.core import setup


AREA=30 #m2
PEOPLE_THRESHOLD = 0.2  # person per m2 so in this example if there are more than 3 person in 10 m2 we give alert
CONFIDENCE_THRESHOLD = 0.4 
SLEEP_TIME = 1           
VIDEO_WIDTH = 640 
VIDEO_HEIGHT = 360
DETECTION_FREQUENCY = 3

class AppDemo(QWidget):
    useWebcam=False
    usePhone=False
    finishProcess= False
    def __init__(self):
        super().__init__()
        self.initUI()
        self.clickableLabel.points_selected.connect(self.on_points_selected)

    def initUI(self):
        self.setWindowTitle('Smart Crowded Control System')

        mainLayout = QVBoxLayout()
   
        videoHeatmapLayout = QGridLayout()
        clickableLayout= QHBoxLayout()

        self.videoLabel = QLabel(self)
        self.videoLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.videoLabel.setMinimumSize(QSize(320, 240))  

        self.clickableLabel = ClickableLabel(self)
        self.clickableLabel.setMaximumSize(1920,1080)
        spacerLeft = QSpacerItem(40, 20, QSizePolicy.Expanding)
        spacerRight = QSpacerItem(40, 20, QSizePolicy.Expanding)
        clickableLayout.addItem(spacerLeft)
        clickableLayout.addWidget(self.clickableLabel)
        clickableLayout.addItem(spacerRight)
        clickableLayout.addStretch(1)
        mainLayout.addLayout(clickableLayout)

        self.transformedVideoLabel = QLabel(self)
        self.transformedVideoLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.transformedVideoLabel.setMinimumSize(QSize(320, 240))  

        self.heatmapLabel = QLabel(self)
        self.heatmapLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.heatmapLabel.setMinimumSize(QSize(320, 240))  

        videoHeatmapLayout.addWidget(self.transformedVideoLabel,0,0)
        videoHeatmapLayout.addWidget(self.videoLabel,0,1)
        videoHeatmapLayout.addWidget(self.heatmapLabel,1,0)

        mainLayout.addLayout(videoHeatmapLayout)
        controlsLayout = QHBoxLayout()

      
        self.label = QLabel('Video Path:')
        controlsLayout.addWidget(self.label)

        self.lineEdit = QLineEdit(self)
        controlsLayout.addWidget(self.lineEdit)

        self.buttonSelect = QPushButton('Choose Video', self)
        self.buttonSelect.clicked.connect(self.select_video)
        controlsLayout.addWidget(self.buttonSelect)

        self.buttonStart = QPushButton('Start Analyze', self)
        self.buttonStart.clicked.connect(self.start_analysis)
        controlsLayout.addWidget(self.buttonStart)
        
        self.buttonSelectWebcam = QPushButton('Use Webcam', self)
        self.buttonSelectWebcam.clicked.connect(self.select_webcam)
        controlsLayout.addWidget(self.buttonSelectWebcam)

        self.buttonPhone = QPushButton('Use Phone', self)
        self.buttonPhone.clicked.connect(self.select_phone)
        controlsLayout.addWidget(self.buttonPhone)

        self.buttonFinish = QPushButton('Finish Process', self)
        self.buttonFinish.clicked.connect(self.finish_process)
        controlsLayout.addWidget(self.buttonFinish)

        mainLayout.addLayout(controlsLayout)

        self.setLayout(mainLayout)
    def finish_process(self) :
        self.finishProcess=True  
    def select_webcam(self):
        self.useWebcam=True 
        
    def select_phone(self):
        self.usePhone=True 
                
    def on_points_selected(self):
        self.clickableLabel.hide()
        video_path = self.lineEdit.text()
        threading.Thread(target=self.analyze_video, args=(video_path,), daemon=True).start()
        
    def select_video(self):
        fname = QFileDialog.getOpenFileName(self, 'Video Seç', '', 'Video files (*.mp4 *.avi)')
        if fname[0]:
            self.lineEdit.setText(fname[0])

    def start_analysis(self):
        
        if self.useWebcam:
            video_path=0
        elif self.usePhone:
            video_path=1
        else:
            video_path = self.lineEdit.text()
        threading.Thread(target=self.analyze_video, args=(video_path,), daemon=True).start()
  
    def convert_to_qimage(self, cv_img):
        """Convert an OpenCV image to QImage"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

    def show_on_label(self, label, qimg):
        """Show QImage on QLabel"""
        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio))        

    def analyze_video(self,video_path):

        decay_rate = 0.95

        max_intensity = 300

        heatmap = np.zeros((300, 200), dtype=np.float32)

        accumulated_heatmap = np.zeros((300, 200), dtype=np.float32)

        classNames=["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
        "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"
        ]

        model = YOLO("yolov8n.pt")

        
        cap = cv2.VideoCapture("Your video path")  #If you want to use webcam feed type 0

        def alert(number): 
            print("[ALERT] People count:", number)


        model = YOLO("yolov8n.pt") 


        
        ret, first_frame = cap.read()
        if not ret:
            print("Failed to grab the first video frame.")
            cap.release()
            return

        
        self.clickableLabel.set_first_frame(first_frame)
        
        reference_points = self.clickableLabel.reference_points


        if len(reference_points) != 4:
            print("You must select exactly 4 points.")
            cap.release()
            exit()


        image_points = np.array(reference_points, dtype='float32')

        #real world coordinates
        real_world_points = np.array([[0, 0], [0, 300], [200, 300], [200, 0]], dtype='float32')
        #homography matrix
        H, status = cv2.findHomography(image_points, real_world_points)

        if H is None or not status.all():
            print("Homography calculation was not successful.")
            cap.release()
            exit()

        # Function to process detections, draw bounding boxes and labels, and return coordinates
        def processResults(results, frame):
            people_coordinates = []  
            for r in results:
                boxes = r.boxes
                peopleCount = 0
                for box in boxes:
                    if box.conf < CONFIDENCE_THRESHOLD:
                        continue
                    cls_id = int(box.cls[0])
                    if classNames[cls_id] == "person":
                        peopleCount += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        center_x, center_y = (x1 + x2) // 2, y2
                        people_coordinates.append((center_x, center_y))  # Store bottom center coordinates
                        # Draw bounding box around detected person
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Put the class name on top of the bounding box
                        text = f"{classNames[cls_id]}: {box.conf[0]:.2f}"
                        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        
                if peopleCount >= PEOPLE_THRESHOLD*AREA: 
                    alert(peopleCount)
            return people_coordinates



        def generate_heatmap(canvas, intensity=10):
            normalized_canvas = cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX)
            incremented_canvas = np.clip(normalized_canvas * intensity, 0, 255)
            heatmap = cv2.applyColorMap(np.uint8(incremented_canvas), cv2.COLORMAP_JET)
            return heatmap

        def update_heatmap(bird_eye_points):
            global heatmap
            heatmap *= decay_rate
            for x, y in bird_eye_points:
                heatmap[int(y), int(x)] += 1  # Increment the count at the location

            #apply Gaussian blur to the heatmap 
            heatmap_blurred = cv2.GaussianBlur(heatmap, (15, 15), 0)

            minVal, maxVal, _, _ = cv2.minMaxLoc(heatmap_blurred)
            if maxVal != 0:
                heatmap_blurred = heatmap_blurred / maxVal
            return heatmap_blurred
        frame_counter = 0
        while cap.isOpened() and not self.finishProcess:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_counter % DETECTION_FREQUENCY == 0:
                
                results = model(frame)
                people_coordinates = processResults(results, frame)
                person_points = np.array(people_coordinates, dtype='float32').reshape(-1, 1, 2)

                # Homography transformation
                if status.all() and len(person_points) > 0:
                    person_points_bird_view = cv2.perspectiveTransform(person_points, H)

                

                    # Update heatmap with bird's eye view points
                    if person_points_bird_view is not None:
                        for point in person_points_bird_view:
                            x, y = point.ravel()
                            if 0 <= y < heatmap.shape[0] and 0 <= x < heatmap.shape[1]:
                                cv2.circle(heatmap, (int(x), int(y)), 3, max_intensity, -1)
                                cv2.circle(accumulated_heatmap, (int(x), int(y)), 3, max_intensity, -1)  # Update accumulated heatmap

                    heatmap = np.clip(heatmap * decay_rate, 0, 300)

                # Rotate and flip the real-time heatmap
                heatmap_rotated = np.rot90(heatmap)
                heatmap_mirrored = np.flipud(heatmap_rotated)
                heatmap_blurred = cv2.GaussianBlur(heatmap_mirrored, (11, 11), 0)
                heatmap_normalized = cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX)
                colored_heatmap = cv2.applyColorMap(np.uint8(heatmap_normalized), cv2.COLORMAP_JET)
                frame_qimg = self.convert_to_qimage(frame)
                heatmap_qimg = self.convert_to_qimage(colored_heatmap)
                self.show_on_label(self.heatmapLabel, heatmap_qimg)

                transformed_frame = cv2.warpPerspective(frame, H, (200, 300)) 

                transformed_frame_rotated = np.rot90(transformed_frame)
                transformed_frame_mirrored = np.flipud(transformed_frame_rotated)
                transformed_frame_qimg = self.convert_to_qimage(transformed_frame_mirrored)
                self.show_on_label(self.transformedVideoLabel, transformed_frame_qimg)
                
            frame_counter += 1

            self.show_on_label(self.videoLabel, frame_qimg)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

        # Normalize the accumulated heatmap based on its maximum value
        max_value = np.max(accumulated_heatmap)
        if max_value > 0:
            accumulated_heatmap_normalized = (accumulated_heatmap / max_value) * 255
        else:
            accumulated_heatmap_normalized = accumulated_heatmap

        accumulated_heatmap_rotated = np.rot90(accumulated_heatmap_normalized)
        accumulated_heatmap_mirrored = np.flipud(accumulated_heatmap_rotated)
        accumulated_heatmap_blurred = cv2.GaussianBlur(accumulated_heatmap_mirrored, (11, 11), 0)
        accumulated_colored_heatmap = cv2.applyColorMap(np.uint8(accumulated_heatmap_blurred), cv2.COLORMAP_JET)
        total_heatmap_qimg = self.convert_to_qimage(accumulated_colored_heatmap)
        self.show_on_label(self.heatmapLabel, total_heatmap_qimg)
        self.show_on_label(self.videoLabel, frame_qimg)

        
        
class ClickableLabel(QLabel):
    points_selected = pyqtSignal()
    def __init__(self, parent=None):
        super(ClickableLabel, self).__init__(parent)
        self.reference_points = []
        self.first_frame = None

    def mousePressEvent(self, event):
        if len(self.reference_points) >= 4:
            return

        if self.first_frame is not None:
            label_width = self.width()
            label_height = self.height()

            frame_height, frame_width, _ = self.first_frame.shape

            x = int(event.x() * (frame_width / label_width))
            y = int(event.y() * (frame_height / label_height))

            self.reference_points.append((x, y))
            self.update_frame_with_points()
        if len(self.reference_points) == 4:
            self.points_selected.emit()

    def set_first_frame(self, frame):
        self.first_frame = frame
        self.update_frame_with_points()

    def update_frame_with_points(self):
        if self.first_frame is not None:
            frame_with_points = self.first_frame.copy()
            for point in self.reference_points:
                cv2.circle(frame_with_points, point, 5, (0, 255, 0), -1)

            frame_qimg = self.convert_to_qimage(frame_with_points)
            self.setPixmap(QPixmap.fromImage(frame_qimg))

    def convert_to_qimage(self, cv_img):
        """Convert an OpenCV image to QImage"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    
app = QApplication(sys.argv)
demo = AppDemo()
demo.show()
sys.exit(app.exec_())


