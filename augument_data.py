import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset folder
DATASET_DIR = "captured_data"
# Augmentation settings
datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.7,1.3],
    fill_mode='nearest'
)

print("Starting dataset augmentation...")

for class_name in os.listdir(DATASET_DIR):

    class_path = os.path.join(DATASET_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    print(f"Processing class: {class_name}")

    for img_name in os.listdir(class_path):

        img_path = os.path.join(class_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (128,128))
        img = np.expand_dims(img, axis=0)

        i = 0

        for batch in datagen.flow(
                img,
                batch_size=1,
                save_to_dir=class_path,
                save_prefix="aug",
                save_format="png"):

            i += 1

            # number of augmented images
            if i >= 40:
                break

print("Dataset augmentation completed!")