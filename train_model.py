import os
import argparse
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical

IMG_SIZE = 128
BATCH_SIZE = 32

def build_model(num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        BatchNormalization(),
        MaxPooling2D(2,2),
        
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train(data_dir, epochs):
    # Detect classes
    class_names = sorted([d for d in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, d))])
    print("Detected Classes:", class_names)
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    
    if num_classes == 0:
        raise ValueError("No class folders found in data_dir!")
    
    # Create datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset='training',
        seed=123,
        shuffle=True
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset='validation',
        seed=123,
        shuffle=True
    )
    
    # Print dataset info
    print("Train batches:", len(train_ds))
    print("Validation batches:", len(val_ds))
    
    # **FIX: Ensure proper one-hot encoding**
    train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    
    # Augmentation layer (applied only to training)
    aug_layer = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomFlip('horizontal')
    ])
    
    def augment_and_normalize(image, label):
        image = aug_layer(image, training=True)
        image = image / 255.0
        # **CRITICAL FIX: Explicit one-hot encoding**
        label = tf.one_hot(tf.cast(label, tf.int32), num_classes)
        return image, label
    
    # Apply augmentation and normalization
    train_ds = train_ds.map(augment_and_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(
        lambda image, label: (image / 255.0, tf.one_hot(tf.cast(label, tf.int32), num_classes)),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Save class mapping (for main.py)
    class_indices = {name: idx for idx, name in enumerate(class_names)}
    with open("classes.json", "w") as f:
        json.dump(class_indices, f)
    print("Class mapping saved:", class_indices)
    
    # Build and train model
    model = build_model(num_classes)
    model.summary()
    
    callbacks = [
        ModelCheckpoint("model-bw.h5", monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)
    ]
    
    print("Starting training...")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    
    # Save complete model
    model.save("model-bw.h5")
    print("✅ Training complete! Saved model-bw.h5 and classes.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hand gesture classifier")
    parser.add_argument("--data_dir", required=True, help="Path to folder with class subfolders")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    args = parser.parse_args()
    
    print(f"Training with data_dir: {args.data_dir}")
    train(args.data_dir, args.epochs)
