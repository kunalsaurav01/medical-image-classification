"""
Hybrid CNN-Transformer Architecture for Medical Image Classification
Author: Research Implementation
Date: October 2025

This implementation combines CNNs for local feature extraction with 
Transformers for global context understanding in medical imaging.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import cv2
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class HybridCNNTransformer:
    """
    Hybrid CNN-Transformer Architecture
    
    Architecture Components:
    1. CNN Feature Extractor (Local Features)
    2. Patch Embedding Layer
    3. Multi-Head Self-Attention (Global Context)
    4. Feed-Forward Network
    5. Classification Head
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=4, 
                 patch_size=16, num_heads=8, num_transformer_blocks=4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_transformer_blocks = num_transformer_blocks
        self.model = None
        self.history = None
        
    def create_cnn_feature_extractor(self, inputs):
        """
        CNN branch for local feature extraction
        Uses residual connections for better gradient flow
        """
        # Block 1
        x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Block 2
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Block 3
        x = layers.Conv2D(256, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(256, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        return x
    
    def patch_embedding(self, x, projection_dim=256):
        """
        Convert CNN features into patches for transformer processing
        """
        # Reshape to patches using Conv2D
        patches = layers.Conv2D(projection_dim, (self.patch_size, self.patch_size), 
                                strides=self.patch_size, padding='valid')(x)
        
        # Get the shape after convolution to calculate num_patches
        # For 56x56 input with patch_size=16, we get 3x3 patches = 9 patches
        # This is computed at build time
        patch_h = 56 // self.patch_size  # Will be 3 for 56x56 input
        patch_w = 56 // self.patch_size  # Will be 3 for 56x56 input
        num_patches = patch_h * patch_w  # 9 patches
        
        # Flatten patches - use explicit reshape with calculated dimensions
        patches = layers.Reshape((num_patches, projection_dim))(patches)
        
        # Create learnable positional embeddings
        # This will be added to each patch
        class AddPositionEmbedding(layers.Layer):
            def __init__(self, num_patches, projection_dim):
                super().__init__()
                self.num_patches = num_patches
                self.projection_dim = projection_dim
                self.position_embedding = layers.Embedding(
                    input_dim=num_patches, 
                    output_dim=projection_dim
                )
            
            def call(self, patches):
                positions = tf.range(start=0, limit=self.num_patches, delta=1)
                embedded_positions = self.position_embedding(positions)
                return patches + embedded_positions
        
        patches = AddPositionEmbedding(num_patches, projection_dim)(patches)
        return patches
    
    def transformer_encoder(self, x, num_heads, ff_dim, dropout=0.1):
        """
        Transformer encoder block with multi-head attention
        """
        # Get the dimension for key_dim
        embed_dim = x.shape[-1]
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim // num_heads
        )(x, x)
        attention_output = layers.Dropout(dropout)(attention_output)
        x1 = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
        
        # Feed-forward network
        ffn_output = layers.Dense(ff_dim, activation='relu')(x1)
        ffn_output = layers.Dropout(dropout)(ffn_output)
        ffn_output = layers.Dense(embed_dim)(ffn_output)
        ffn_output = layers.Dropout(dropout)(ffn_output)
        x2 = layers.LayerNormalization(epsilon=1e-6)(x1 + ffn_output)
        
        return x2
    
    def build_model(self):
        """
        Build the complete hybrid architecture
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # CNN Feature Extraction
        cnn_features = self.create_cnn_feature_extractor(inputs)
        
        # Patch Embedding
        projection_dim = 256
        patches = self.patch_embedding(cnn_features, projection_dim)
        
        # Transformer Blocks
        x = patches
        for _ in range(self.num_transformer_blocks):
            x = self.transformer_encoder(
                x, 
                num_heads=self.num_heads, 
                ff_dim=projection_dim * 2
            )
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Classification Head
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = models.Model(inputs=inputs, outputs=outputs)
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer and loss function
        """
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the model with callbacks
        """
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
        )
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return self.history


class DataProcessor:
    """
    Data preprocessing and augmentation pipeline
    """
    
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
        
    def load_and_preprocess(self, image_path):
        """
        Load and preprocess a single image
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.astype('float32') / 255.0
        return img
    
    def augment_data(self):
        """
        Create data augmentation pipeline
        """
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2),
        ])
        return data_augmentation
    
    def create_balanced_dataset(self, X, y, method='oversample'):
        """
        Handle class imbalance using oversampling or undersampling
        """
        from collections import Counter
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        
        # Flatten images for SMOTE
        X_flat = X.reshape(X.shape[0], -1)
        y_labels = np.argmax(y, axis=1)
        
        if method == 'oversample':
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X_flat, y_labels)
        elif method == 'undersample':
            rus = RandomUnderSampler(random_state=42)
            X_balanced, y_balanced = rus.fit_resample(X_flat, y_labels)
        
        # Reshape back
        X_balanced = X_balanced.reshape(-1, *self.img_size, 3)
        y_balanced = keras.utils.to_categorical(y_balanced, num_classes=y.shape[1])
        
        return X_balanced, y_balanced


class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization
    """
    
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
        
    def plot_training_history(self, history, save_path='training_history.png'):
        """
        Plot training and validation metrics
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(history.history['loss'], label='Train Loss')
        axes[1].plot(history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # AUC
        axes[2].plot(history.history['auc'], label='Train AUC')
        axes[2].plot(history.history['val_auc'], label='Val AUC')
        axes[2].set_title('Model AUC', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('AUC')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrix(self, y_true, y_pred, save_path='confusion_matrix.png'):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_roc_curves(self, y_true, y_pred_proba, save_path='roc_curves.png'):
        """
        Plot ROC curves for all classes
        """
        n_classes = len(self.class_names)
        
        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, color in zip(range(n_classes), colors[:n_classes]):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Multi-Class Classification', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_classification_report(self, y_true, y_pred, save_path='classification_report.txt'):
        """
        Generate and save classification report
        """
        report = classification_report(y_true, y_pred, 
                                      target_names=self.class_names,
                                      digits=4)
        
        print("\nClassification Report:")
        print("=" * 60)
        print(report)
        
        with open(save_path, 'w') as f:
            f.write(report)
        
        return report


def compare_with_baseline_models(X_train, y_train, X_test, y_test, input_shape, num_classes):
    """
    Compare hybrid model with baseline models
    """
    results = {}
    
    # 1. Simple CNN
    print("\n" + "="*60)
    print("Training Simple CNN...")
    print("="*60)
    
    cnn_model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    cnn_model.compile(optimizer='adam', 
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
    
    cnn_model.fit(X_train, y_train, epochs=20, batch_size=32, 
                  validation_split=0.2, verbose=0)
    
    cnn_loss, cnn_acc = cnn_model.evaluate(X_test, y_test, verbose=0)
    results['Simple CNN'] = {'accuracy': cnn_acc, 'loss': cnn_loss}
    print(f"Simple CNN - Accuracy: {cnn_acc:.4f}, Loss: {cnn_loss:.4f}")
    
    # 2. ResNet-like Model
    print("\n" + "="*60)
    print("Training ResNet-like Model...")
    print("="*60)
    
    base_model = keras.applications.ResNet50(
        include_top=False, 
        weights=None,
        input_shape=input_shape
    )
    
    resnet_model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    resnet_model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
    
    resnet_model.fit(X_train, y_train, epochs=20, batch_size=32,
                    validation_split=0.2, verbose=0)
    
    resnet_loss, resnet_acc = resnet_model.evaluate(X_test, y_test, verbose=0)
    results['ResNet-like'] = {'accuracy': resnet_acc, 'loss': resnet_loss}
    print(f"ResNet-like - Accuracy: {resnet_acc:.4f}, Loss: {resnet_loss:.4f}")
    
    return results


# Main execution function
def main():
    """
    Main execution pipeline
    """
    print("\n" + "="*80)
    print("HYBRID CNN-TRANSFORMER MEDICAL IMAGE CLASSIFICATION")
    print("="*80 + "\n")
    
    # Set parameters
    IMG_SIZE = (224, 224)
    NUM_CLASSES = 4
    BATCH_SIZE = 32
    EPOCHS = 50
    
    # Class names (modify based on your dataset)
    CLASS_NAMES = ['Normal', 'Bacterial', 'Viral', 'COVID-19']
    
    print("Step 1: Data Loading and Preprocessing")
    print("-" * 60)
    
    # For demonstration, create synthetic data
    # Replace this with actual data loading
    print("Creating synthetic dataset for demonstration...")
    X_data = np.random.rand(1000, 224, 224, 3).astype('float32')
    y_data = keras.utils.to_categorical(
        np.random.randint(0, NUM_CLASSES, 1000), 
        num_classes=NUM_CLASSES
    )
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_data, y_data, test_size=0.3, random_state=42, stratify=np.argmax(y_data, axis=1)
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=np.argmax(y_temp, axis=1)
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Build and train hybrid model
    print("\n" + "="*80)
    print("Step 2: Building Hybrid CNN-Transformer Model")
    print("-" * 60)
    
    hybrid_model = HybridCNNTransformer(
        input_shape=(*IMG_SIZE, 3),
        num_classes=NUM_CLASSES,
        patch_size=16,
        num_heads=8,
        num_transformer_blocks=4
    )
    
    model = hybrid_model.build_model()
    print(f"Total parameters: {model.count_params():,}")
    
    hybrid_model.compile_model(learning_rate=0.001)
    
    print("\nStep 3: Training Model")
    print("-" * 60)
    
    history = hybrid_model.train(
        X_train, y_train, X_val, y_val,
        epochs=EPOCHS, batch_size=BATCH_SIZE
    )
    
    # Evaluation
    print("\n" + "="*80)
    print("Step 4: Model Evaluation")
    print("-" * 60)
    
    evaluator = ModelEvaluator(model, CLASS_NAMES)
    
    # Plot training history
    evaluator.plot_training_history(history)
    
    # Predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Confusion matrix
    evaluator.plot_confusion_matrix(y_true, y_pred)
    
    # ROC curves
    evaluator.plot_roc_curves(y_test, y_pred_proba)
    
    # Classification report
    evaluator.generate_classification_report(y_true, y_pred)
    
    # Compare with baseline
    print("\n" + "="*80)
    print("Step 5: Comparison with Baseline Models")
    print("-" * 60)
    
    baseline_results = compare_with_baseline_models(
        X_train, y_train, X_test, y_test,
        (*IMG_SIZE, 3), NUM_CLASSES
    )
    
    # Add hybrid model results
    test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
    baseline_results['Hybrid CNN-Transformer'] = {
        'accuracy': test_acc, 
        'loss': test_loss
    }
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    models = list(baseline_results.keys())
    accuracies = [baseline_results[m]['accuracy'] for m in models]
    
    bars = plt.bar(models, accuracies, color=['#3498db', '#e74c3c', '#2ecc71'])
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Comparison - Test Accuracy', fontsize=16, fontweight='bold')
    plt.ylim([0, 1])
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*80)
    print("Training and Evaluation Complete!")
    print("="*80)
    print(f"\nHybrid Model Final Results:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    # Save model
    model.save('hybrid_cnn_transformer_model.h5')
    print("\nModel saved as 'hybrid_cnn_transformer_model.h5'")


if __name__ == "__main__":

    main()
