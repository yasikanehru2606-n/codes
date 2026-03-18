#!/usr/bin/env python3
"""
Diagnostic script to debug the sign language recognition system
Run this to understand what's going wrong
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp

# Get the directory
script_dir = os.path.dirname(os.path.abspath(__file__))

print("=" * 70)
print("SIGN LANGUAGE RECOGNITION SYSTEM - DIAGNOSTIC REPORT")
print("=" * 70)

# 1. Check model
print("\n1. MODEL LOADING STATUS")
print("-" * 70)
model_path = os.path.join(script_dir, "model.h5")
print(f"   Looking for: {model_path}")
print(f"   File exists: {os.path.exists(model_path)}")

if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"   ✓ Model loaded successfully!")
        print(f"     Input shape: {model.input_shape}")
        print(f"     Output shape: {model.output_shape}")
        print(f"     Number of classes: {model.output_shape[-1]}")
        
        # Test prediction with dummy input
        print("\n   Testing model prediction with dummy input (128x128x1):")
        dummy_input = np.random.rand(1, 128, 128, 1).astype(np.float32)
        dummy_pred = model.predict(dummy_input, verbose=0)[0]
        top_idx = int(np.argmax(dummy_pred))
        top_conf = float(np.max(dummy_pred))
        print(f"     Top prediction index: {top_idx}")
        print(f"     Top confidence: {top_conf:.4f}")
        print(f"     ✓ Model is working!")
        
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        model = None
else:
    print(f"   ✗ Model file not found!")
    print(f"     System will run in DEMO MODE")
    model = None

# 2. Check labels
print("\n2. LABEL MAPPING STATUS")
print("-" * 70)

def load_label_map():
    for filename in ("labels.json", "classes.json"):
        path = os.path.join(script_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r") as f:
                data = json.load(f)

            if isinstance(data, dict):
                if all(isinstance(v, int) for v in data.values()):
                    return {int(v): k for k, v in data.items()}
                if all(k.isdigit() for k in data.keys()):
                    return {int(k): v for k, v in data.items()}

            if isinstance(data, list):
                return {i: v for i, v in enumerate(data)}

            print(f"Unsupported format in {path}")
        except Exception as e:
            print(f"Error reading {path}: {e}")

    return {i: chr(ord('A') + i) for i in range(26)}

idx_to_label = load_label_map()
print(f"   Loaded {len(idx_to_label)} labels")
print(f"   First 10 labels: {[idx_to_label[i] for i in range(min(10, len(idx_to_label)))]}")

if len(idx_to_label) > 0:
    print(f"   ✓ Labels loaded successfully!")
else:
    print(f"   ✗ No labels found!")

# 3. Check MediaPipe
print("\n3. MEDIAPIPE STATUS")
print("-" * 70)

try:
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    print(f"   ✓ MediaPipe initialized successfully!")
except Exception as e:
    print(f"   ✗ Error initializing MediaPipe: {e}")

# 4. Check camera
print("\n4. CAMERA STATUS")
print("-" * 70)

try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"   ✓ Camera is working!")
            print(f"     Frame shape: {frame.shape}")
        else:
            print(f"   ✗ Camera opened but can't read frames")
        cap.release()
    else:
        print(f"   ✗ Camera not available")
except Exception as e:
    print(f"   ✗ Error accessing camera: {e}")

# 5. Check data files
print("\n5. DATA FILES STATUS")
print("-" * 70)

filtered_data = os.path.join(script_dir, "filtered_data")
alphabet = os.path.join(script_dir, "alphabet")

print(f"   Filtered data folder: {os.path.exists(filtered_data)}")
print(f"   Alphabet folder: {os.path.exists(alphabet)}")

if os.path.exists(filtered_data):
    webp_count = sum(1 for root, dirs, files in os.walk(filtered_data) 
                     for f in files if f.endswith('.webp'))
    print(f"   WebP files found: {webp_count}")

# 6. Summary
print("\n" + "=" * 70)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 70)

if model is None:
    print("\n⚠ MODEL NOT LOADED")
    print("  The system is running in DEMO MODE")
    print("  → Make sure model.h5 exists in the project directory")
    print("  → Verify the model file is not corrupted")
else:
    print("\n✓ MODEL LOADED")
    print("  The actual trained model is being used")

if len(idx_to_label) == 0:
    print("\n✗ NO LABELS FOUND")
    print("  → Create labels.json or classes.json file")
else:
    print(f"\n✓ LABELS LOADED ({len(idx_to_label)} classes)")

print("\n" + "=" * 70)
print("If you see only 'A' in the character label:")
print("=" * 70)
print("\n1. Check if model.h5 is loading (should see 'Model loaded successfully')")
print("2. Check console output while making signs:")
print("   - Look for '[LOW CONF]' messages → confidence too low")
print("   - Look for '[AMBIGUOUS]' messages → needs more frames")
print("   - Look for '✓ STABLE' messages → character accepted")
print("3. If all predictions show '[LOW CONF]':")
print("   → Model confidence is low, may need retraining")
print("   → Or temporary fix: lower confidence threshold from 0.75 to 0.60")
print("\n" + "=" * 70)
