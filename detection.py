import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog

class detection:
    
    @staticmethod
    def detect_face(image_path, img_size=(92, 112), scaleFactor=1.1, minNeighbors=5):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image {image_path}")
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

        if len(faces) == 0:
            # print("No faces found.")
            return None

        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, img_size)
        return face_resized

def select_and_detect_face():
    app = QApplication(sys.argv)  
    file_path, _ = QFileDialog.getOpenFileName(
        None, "Select an Image File", "", "Image files (*.jpg *.jpeg *.png)"
    )

    if not file_path:
        print("No file selected.")
        return

    print(f"Selected file: {file_path}")
    face = detection.detect_face(file_path)

    if face is None:
        print("No face detected.")
        return

    plt.imshow(face, cmap='gray')
    plt.title("Detected Face")
    plt.axis("off")
    plt.show()

## for testing (usually in pics no. 9,10, and 14 in training directory no faces can be detected, in testing directory all work)
# select_and_detect_face()
