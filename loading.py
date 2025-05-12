import numpy as np
from sklearn.model_selection import train_test_split
import os
from detection import detection  
import cv2
import re
class loading:
    @staticmethod
    def extract_subject_id(filename):
         # Extracts the first numeric sequence in the filename
        match = re.match(r"(\d+)", filename)
        if match:
            return match.group(1)
        else:
            return filename  # fallback if no digits found

    @staticmethod
    def load_dataset(train_dir, test_dir, image_size=(100, 100)):
        X_train, y_train = [], []
        X_test, y_test = [], []
        x_test_paths = []

        label_map = {}
        label_counter = 0

        def process_folder(folder, is_train=True):
            nonlocal label_counter
            X, y, paths = [], [], []

            for fname in sorted(os.listdir(folder)):
                if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue

                subject_id = loading.extract_subject_id(fname) 
                if subject_id not in label_map:
                    label_map[subject_id] = label_counter
                    label_counter += 1
                label = label_map[subject_id]

                img_path = os.path.join(folder, fname)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Failed to read {img_path}")
                    continue

                face = detection.detect_face(img_path)  # Use detection method to detect faces
                if face is None:
                    continue

                face_resized = cv2.resize(face, image_size)
                X.append(face_resized)
                y.append(label)
                paths.append(img_path)

            return X, y, paths

        # Process the training and testing directories
        X_train, y_train, _ = process_folder(train_dir, is_train=True)
        X_test, y_test, x_test_paths = process_folder(test_dir, is_train=False)

        print(f"Loaded {len(X_train)} training faces.")
        print(f"Loaded {len(X_test)} testing faces.")
        return (np.array(X_train), np.array(y_train),
                np.array(X_test), np.array(y_test),
                label_map, x_test_paths)


# testing (very very slow)
# Add your local paths for the dataset
# dataset_path = "C:\\Users\\DR.Mahmoud\\Downloads\\PCA_data\\FEI_faces"
# testing_path = "C:\\Users\\DR.Mahmoud\\Downloads\\PCA_data\\FEI_testing"
#
# # Load data
# X_train, y_train, X_test, y_test, label_map, X_test_files = loading.load_dataset(dataset_path, testing_path)
#
# print(f"Train samples: {len(X_train)}")
# print(f"Test samples: {len(X_test)}")
# print(f"Subjects found: {len(label_map)}")
# print(f"Example label map: {label_map}")
# print(X_test_files)
