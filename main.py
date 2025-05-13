import sys
import os
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QFileDialog, QMessageBox, QPushButton, QSlider, QSpinBox, \
    QScrollArea, QProgressBar
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import uic
from ROC import ROCCurve
from detection import detection
from CustomPCA import Custom_PCA
import matplotlib.pyplot as plt
import io
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("PCA_Face_Analysis_Responsive.ui")
        
        # Initialize variables
        self.current_image_path = None
        self.detected_face = None
        self.transformed_face = None
        self.models_loaded = False
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models")

        self.btnLoadImage = self.ui.findChild(QPushButton, "btnLoadImage")
        self.btnDetectFace = self.ui.findChild(QPushButton, "btnFaceDetection")
        self.apply_pca = self.ui.findChild(QPushButton, "apply_pca")
        self.btnMatch = self.ui.findChild(QPushButton, "btnFaceMatching")
        self.btnROC = self.ui.findChild(QPushButton, "btnROC")

        self.scale_factor = self.ui.findChild(QSlider, "scale_factor_slider")
        self.min_neighbors = self.ui.findChild(QSpinBox, "min_neighbors")
        
        # Connect UI elements to functions
        self.btnLoadImage.clicked.connect(self.load_image)
        self.btnDetectFace.clicked.connect(self.detect_face)
        self.apply_pca.clicked.connect(self.apply_pca_transform)
        self.btnMatch.clicked.connect(self.match_face)
        self.btnROC.clicked.connect(self.plot_roc_curve)
        
        # Set default values for face detection parameters
        self.scale_factor.setValue(11)  # 1.1 * 10
        self.min_neighbors.setValue(5)

        self.eigenface1 = self.ui.findChild(QLabel, "eigenface1")
        self.eigenface2 = self.ui.findChild(QLabel, "eigenface2")
        self.eigenface3 = self.ui.findChild(QLabel, "eigenface3")
        self.eigenface4 = self.ui.findChild(QLabel, "eigenface4")
        self.eigenface5 = self.ui.findChild(QLabel, "eigenface5")

        self.InputImage = self.ui.findChild(QLabel, "InputImage")
        self.OutputImage = self.ui.findChild(QLabel, "OutputImage")

        self.match_result = self.ui.findChild(QLabel, "match_result")
        self.match_confidence = self.ui.findChild(QProgressBar, "match_confidence")
        
        # Load models
        self.load_models()
        
        # Set window title
        self.setWindowTitle("Face Recognition with PCA")
        self.setCentralWidget(self.ui)
        self.resize(1360, 768)
    
    def load_models(self):
        try:
            # Load PCA model
            self.pca = Custom_PCA()
            self.pca.load(os.path.join(self.models_dir, "pca_model.npz"))
            
            # Load KNN model
            knn_data = np.load(os.path.join(self.models_dir, "knn_model.npz"), allow_pickle=True)
            self.knn = KNeighborsClassifier(n_neighbors=int(knn_data['n_neighbors']), metric=str(knn_data['metric']))
            
            # Load scaler model
            scaler_data = np.load(os.path.join(self.models_dir, "scaler_model.npz"), allow_pickle=True)
            self.scaler = StandardScaler()
            self.scaler.mean_ = scaler_data['mean']
            self.scaler.scale_ = scaler_data['scale']
            
            # Load label map
            label_data = np.load(os.path.join(self.models_dir, "label_map.npz"), allow_pickle=True)
            self.label_map = label_data['label_map'][()]
            self.inv_label_map = {v: k for k, v in self.label_map.items()}
            
            # Load transformed training data
            self.X_train_pca = np.load(os.path.join(self.models_dir, "X_train_pca.npz"))['X_train_pca']
            self.y_train = np.load(os.path.join(self.models_dir, "y_train.npz"))['y_train']

            self.knn.fit(self.X_train_pca, self.y_train)
            
            # Display eigenfaces
            self.display_eigenfaces()
            
            self.models_loaded = True
            self.statusBar().showMessage("Models loaded successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load models: {str(e)}")
            self.statusBar().showMessage("Failed to load models")
    
    def display_eigenfaces(self):
        # Get the first 5 eigenfaces
        eigenfaces = self.pca.components_.T[:5]
        
        # Normalize and reshape eigenfaces for display
        for i, eigenface in enumerate(eigenfaces, 1):
            # Reshape eigenface to original image dimensions
            eigenface_img = eigenface.reshape(100, 100)
            
            # Normalize to 0-255 range
            eigenface_img = ((eigenface_img - eigenface_img.min()) / 
                            (eigenface_img.max() - eigenface_img.min()) * 255).astype(np.uint8)
            
            # Convert to QImage and display
            h, w = eigenface_img.shape
            q_img = QImage(eigenface_img.data, w, h, w, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_img)
            
            # Get the corresponding eigenface label and set the pixmap
            eigenface_label = getattr(self, f"eigenface{i}")
            eigenface_label.setPixmap(pixmap)
            eigenface_label.setScaledContents(True)
    
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select an Image File", "", "Image files (*.jpg *.jpeg *.png)"
        )
        
        if not file_path:
            return
        
        self.current_image_path = file_path
        
        # Load and display the image
        img = cv2.imread(file_path)
        if img is None:
            QMessageBox.critical(self, "Error", f"Could not load image: {file_path}")
            return
        
        # Convert BGR to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        
        # Convert to QImage and display
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Set image to QLabel
        self.InputImage.setPixmap(pixmap)
        self.InputImage.setScaledContents(True)
        
        # Clear output image and reset variables
        self.OutputImage.clear()
        self.detected_face = None
        self.transformed_face = None
        self.match_result.clear()
        self.match_confidence.setValue(0)
        
        self.statusBar().showMessage(f"Loaded image: {os.path.basename(file_path)}")
    
    def detect_face(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return
        
        # Get face detection parameters
        scale_factor = self.scale_factor.value() / 10.0  # Convert from int to float (e.g., 11 -> 1.1)
        min_neighbors = self.min_neighbors.value()
        
        # Detect face
        self.detected_face = detection.detect_face(
            self.current_image_path, 
            img_size=(92, 112),  # Match the size used in training
            scaleFactor=scale_factor, 
            minNeighbors=min_neighbors
        )
        
        if self.detected_face is None:
            QMessageBox.warning(self, "Warning", "No face detected in the image.")
            return
        
        # Display detected face
        h, w = self.detected_face.shape
        q_img = QImage(self.detected_face.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        
        self.OutputImage.setPixmap(pixmap)
        self.OutputImage.setScaledContents(True)
        
        self.statusBar().showMessage("Face detected successfully")
    
    def apply_pca_transform(self):
        if not self.models_loaded:
            QMessageBox.warning(self, "Warning", "Models not loaded. Please restart the application.")
            return
        
        if self.detected_face is None:
            QMessageBox.warning(self, "Warning", "Please detect a face first.")
            return
        
        # Update number of components if changed
        # num_components = self.num_eigenfaces.value()
        # if num_components != self.pca.n_components:
        #     # We would need to reload the PCA model and adjust it
        #     # For simplicity, we'll just inform the user
        #     QMessageBox.information(self, "Information",
        #                           f"Using {self.pca.n_components} components from the loaded model. "
        #                           f"To change this, you would need to retrain the model.")
        
        # Flatten the face image and reshape it for PCA
        detected_face = cv2.resize(self.detected_face, (100, 100))
        face_vector = detected_face.flatten().reshape(1, -1)
        
        # Apply the same preprocessing as during training
        face_vector_scaled = self.scaler.transform(face_vector)
        
        # Transform using PCA
        self.transformed_face = self.pca.transform(face_vector_scaled)
        
        # Reconstruct the face for visualization
        reconstructed_face = self.pca.inverse_transform(self.transformed_face)
        reconstructed_face = self.scaler.inverse_transform(reconstructed_face)
        
        # Reshape and normalize for display
        reconstructed_img = reconstructed_face.reshape(100, 100)
        reconstructed_img = ((reconstructed_img - reconstructed_img.min()) / 
                           (reconstructed_img.max() - reconstructed_img.min()) * 255).astype(np.uint8)
        
        # Display reconstructed face
        h, w = reconstructed_img.shape
        q_img = QImage(reconstructed_img.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        
        self.OutputImage.setPixmap(pixmap)
        self.OutputImage.setScaledContents(True)
        
        # Find the top 5 eigenfaces with highest weights
        weights = self.transformed_face[0]  # Get the weights from the transformed face
        
        # Get indices of eigenfaces sorted by absolute weight values (highest first)
        top_indices = np.argsort(np.abs(weights))[::-1][:5]  # Top 5 indices
        top_weights = weights[top_indices]  # Corresponding weights
        
        # Get all eigenfaces
        all_eigenfaces = self.pca.components_.T  # Shape: (n_features, n_components)
        
        # Display the top 5 eigenfaces with highest weights
        for i, (idx, weight) in enumerate(zip(top_indices, top_weights), 1):
            # Get the eigenface with the corresponding index
            eigenface = all_eigenfaces[idx, :]
            
            # Reshape eigenface to original image dimensions
            eigenface_img = eigenface.reshape(100, 100)
            
            # Normalize to 0-255 range
            eigenface_img = ((eigenface_img - eigenface_img.min()) / 
                            (eigenface_img.max() - eigenface_img.min()) * 255).astype(np.uint8)
            
            # Convert to QImage and display
            h, w = eigenface_img.shape
            q_img = QImage(eigenface_img.data, w, h, w, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_img)
            
            # Get the corresponding eigenface label and set the pixmap
            eigenface_label = getattr(self, f"eigenface{i}")
            eigenface_label.setPixmap(pixmap)
            eigenface_label.setScaledContents(True)
        
        self.statusBar().showMessage(f"PCA transformation applied - Top 5 eigenfaces displayed by weight importance")
    
    def match_face(self):
        if not self.models_loaded:
            QMessageBox.warning(self, "Warning", "Models not loaded. Please restart the application.")
            return
        
        if self.transformed_face is None:
            QMessageBox.warning(self, "Warning", "Please apply PCA transformation first.")
            return
        
        # Use KNN to find the closest match
        distances, indices = self.knn.kneighbors(self.transformed_face)
        
        # Get the predicted label and distance
        closest_idx = indices[0][0]
        closest_distance = distances[0][0]
        predicted_label = self.y_train[closest_idx]
        
        # Convert to subject ID using inverse label map
        subject_id = self.inv_label_map.get(predicted_label, "Unknown")
        
        # Calculate confidence (inverse of distance, normalized)
        max_distance = 5000  # Arbitrary max distance for normalization
        confidence = max(0, min(100, 100 * (1 - closest_distance / max_distance)))
        
        # Display match result
        self.match_result.setText(f"Subject ID: {subject_id}\nDistance: {closest_distance:.2f}")
        self.match_confidence.setValue(int(confidence))
        
        self.statusBar().showMessage(f"Matched to Subject ID: {subject_id} with {confidence:.1f}% confidence")
    
    def plot_roc_curve(self):
        if not self.models_loaded:
            QMessageBox.warning(self, "Warning", "Models not loaded. Please restart the application.")
            return
        
        # For ROC curve, we need true labels and scores
        # We'll use distances to the correct class vs. incorrect classes
        
        # Get unique labels
        unique_labels = np.unique(self.y_train)
        
        # Prepare data for ROC curve
        y_true = []
        y_scores = []
        
        # For each sample in the training set
        for i, sample in enumerate(self.X_train_pca):
            true_label = self.y_train[i]
            
            # Calculate distances to all training samples
            distances = np.sqrt(np.sum((self.X_train_pca - sample)**2, axis=1))
            
            # For each possible label
            for label in unique_labels:
                # 1 if this is the correct label, 0 otherwise
                y_true.append(1 if label == true_label else 0)
                
                # Use negative distance as score (closer = higher score)
                # Get minimum distance to samples of this label
                label_indices = np.where(self.y_train == label)[0]
                min_distance = np.min(distances[label_indices])
                y_scores.append(-min_distance)
        
        # Compute ROC curve
        roc = ROCCurve()
        roc.compute_roc(np.array(y_true), np.array(y_scores))
        
        # Plot ROC on matplotlib
        fig, ax = plt.subplots(figsize=(5, 4))
        roc.plot_roc_curve(ax=ax, title="ROC Curve for Face Recognition")
        
        # Save to a buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        
        # Load image from buffer
        image = QImage()
        image.loadFromData(buf.read(), 'PNG')
        pixmap = QPixmap.fromImage(image)
        
        # Set image to QLabel
        self.OutputImage.setPixmap(pixmap)
        self.OutputImage.setScaledContents(True)
        
        plt.close(fig)
        
        self.statusBar().showMessage(f"ROC Curve plotted with AUC: {roc.auc:.3f}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())
