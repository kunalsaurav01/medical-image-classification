"""
Data Loader for Medical Image Classification
Supports multiple dataset formats and sources
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
from tqdm import tqdm
import kaggle


class MedicalImageDataLoader:
    """
    Comprehensive data loader for medical imaging datasets
    Supports: Chest X-Ray, CT scans, MRI, etc.
    """
    
    def __init__(self, data_path, img_size=(224, 224)):
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.images = []
        self.labels = []
        self.class_names = []
        
    def download_chest_xray_dataset(self):
        """
        Download Chest X-Ray dataset from Kaggle
        Dataset: COVID-19 Radiography Database or similar
        
        Before running:
        1. Install kaggle: pip install kaggle
        2. Create API token from kaggle.com/account
        3. Place kaggle.json in ~/.kaggle/
        """
        try:
            print("Downloading dataset from Kaggle...")
            # Example: COVID-19 Radiography Database
            kaggle.api.dataset_download_files(
                'tawsifurrahman/covid19-radiography-database',
                path='./data',
                unzip=True
            )
            print("Dataset downloaded successfully!")
            return True
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please download manually from Kaggle")
            return False
    
    def load_from_directory(self, data_dir):
        """
        Load images from directory structure:
        data_dir/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img1.jpg
                img2.jpg
        """
        data_dir = Path(data_dir)
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Directory {data_dir} not found!")
        
        # Get class names from subdirectories
        self.class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
        print(f"Found {len(self.class_names)} classes: {self.class_names}")
        
        # Load images
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = data_dir / class_name
            image_files = list(class_dir.glob('*.jpg')) + \
                         list(class_dir.glob('*.jpeg')) + \
                         list(class_dir.glob('*.png'))
            
            print(f"Loading {len(image_files)} images from {class_name}...")
            
            for img_path in tqdm(image_files, desc=class_name):
                try:
                    # Load and preprocess image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)
                    img = img.astype('float32') / 255.0
                    
                    self.images.append(img)
                    self.labels.append(class_idx)
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        
        print(f"\nTotal images loaded: {len(self.images)}")
        print(f"Image shape: {self.images.shape}")
        print(f"Label distribution: {np.bincount(self.labels)}")
        
        return self.images, self.labels
    
    def load_from_csv(self, csv_path, image_col='filename', label_col='label', image_dir='images'):
        """
        Load images from CSV metadata file
        CSV format:
        filename,label
        img1.jpg,0
        img2.jpg,1
        """
        df = pd.read_csv(csv_path)
        image_dir = Path(image_dir)
        
        print(f"Loading {len(df)} images from CSV...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                img_path = image_dir / row[image_col]
                
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
                img = img.astype('float32') / 255.0
                
                self.images.append(img)
                self.labels.append(row[label_col])
                
            except Exception as e:
                continue
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        
        # Get unique class names
        self.class_names = sorted(list(set(df[label_col].unique())))
        
        return self.images, self.labels
    
    def create_tf_dataset(self, X, y, batch_size=32, shuffle=True, augment=False):
        """
        Create TensorFlow dataset with optional augmentation
        """
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        if augment:
            augmentation = keras.Sequential([
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomRotation(0.2),
                keras.layers.RandomZoom(0.2),
                keras.layers.RandomContrast(0.2),
            ])
            dataset = dataset.map(
                lambda x, y: (augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_class_weights(self, labels):
        """
        Calculate class weights for imbalanced datasets
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        
        class_weight_dict = dict(enumerate(class_weights))
        print("\nClass weights:", class_weight_dict)
        
        return class_weight_dict


def prepare_dataset_for_training(data_path, test_size=0.2, val_size=0.1):
    """
    Complete pipeline to prepare dataset for training
    
    Args:
        data_path: Path to dataset directory
        test_size: Proportion of test set
        val_size: Proportion of validation set from remaining data
    
    Returns:
        Dictionary containing train, validation, and test splits
    """
    print("="*80)
    print("PREPARING DATASET FOR TRAINING")
    print("="*80)
    
    # Initialize loader
    loader = MedicalImageDataLoader(data_path)
    
    # Load data
    X, y = loader.load_from_directory(data_path)
    
    # Convert labels to categorical
    num_classes = len(loader.class_names)
    y_categorical = keras.utils.to_categorical(y, num_classes=num_classes)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_categorical, test_size=test_size, random_state=42, stratify=y
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42,
        stratify=np.argmax(y_temp, axis=1)
    )
    
    print("\n" + "="*80)
    print("Dataset Split Summary:")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print("="*80)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'class_names': loader.class_names,
        'num_classes': num_classes
    }


# Example usage
if __name__ == "__main__":
    # Example 1: Load from directory structure
    data_path = "./data/chest_xray"
    
    try:
        dataset = prepare_dataset_for_training(data_path)
        print("\nDataset loaded successfully!")
        print(f"Classes: {dataset['class_names']}")
    except Exception as e:
        print(f"Error: {e}")
        print("\nCreating synthetic dataset for testing...")
        
        # Create synthetic data
        X_train = np.random.rand(800, 224, 224, 3).astype('float32')
        y_train = keras.utils.to_categorical(np.random.randint(0, 4, 800), 4)
        X_val = np.random.rand(100, 224, 224, 3).astype('float32')
        y_val = keras.utils.to_categorical(np.random.randint(0, 4, 100), 4)
        X_test = np.random.rand(100, 224, 224, 3).astype('float32')
        y_test = keras.utils.to_categorical(np.random.randint(0, 4, 100), 4)
        
        dataset = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'class_names': ['Normal', 'Bacterial', 'Viral', 'COVID-19'],
            'num_classes': 4
        }
        print("Synthetic dataset created!")