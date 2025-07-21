import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_dir, img_size=(150, 150)):
    images = []
    labels = []
    
    for label in ['no_tumor', 'tumor']:
        path = os.path.join(data_dir, label)
        class_num = 0 if label == 'no_tumor' else 1
        
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_array = cv2.resize(img_array, img_size)
                images.append(resized_array)
                labels.append(class_num)
            except Exception as e:
                print(f"Error loading {img}: {e}")
                
    return np.array(images), np.array(labels)

def preprocess_data(X, y):
    # Normalize pixel values
    X = X / 255.0
    
    # Reshape for CNN input
    X = X.reshape(-1, X.shape[1], X.shape[2], 1)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_val, y_train, y_val
