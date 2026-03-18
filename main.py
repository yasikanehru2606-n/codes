from cProfile import label

import warnings
warnings.filterwarnings('ignore', message='.*SymbolDatabase.GetPrototype.*')

import customtkinter as ctk
import csv
import tkinter as tk
from tkinter import Label
import cv2
import tensorflow as tf
from PIL import Image, ImageTk
import mediapipe as mp
import itertools
import copy
from datetime import datetime
import os
import numpy as np
import PIL
from collections import deque
import json

# Function to calculate the landmark points from an image
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Iterate over each landmark and convert its coordinates
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

# Function to preprocess landmark data
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load labels (labels.json preferred, fallback to classes.json)
def _load_label_map():
    for filename in ("labels.json", "classes.json"):
        path = os.path.join(script_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r") as f:
                data = json.load(f)

            if isinstance(data, dict):
                # {"A": 0, "B": 1}
                if all(isinstance(v, int) for v in data.values()):
                    return {int(v): k for k, v in data.items()}
                # {"0": "A", "1": "B"}
                if all(k.isdigit() for k in data.keys()):
                    return {int(k): v for k, v in data.items()}

            if isinstance(data, list):
                return {i: v for i, v in enumerate(data)}

            print(f"Unsupported label file format: {path}")
        except Exception as e:
            print(f"Failed to load label file {path}: {e}")

    return {i: chr(ord('A') + i) for i in range(26)}

idx_to_label = _load_label_map()
print(f"Loaded label mapping ({len(idx_to_label)} entries)")

# Load the keypoint classifier model (expects 128x128x1 grayscale hand image)
try:
    model_path = os.path.join(script_dir, "model.h5")
    if os.path.exists(model_path):
        keypoint_model = tf.keras.models.load_model(model_path)
        print(f"INFO: Loaded model from {model_path}")
        print(f"      Input shape: {keypoint_model.input_shape}")
        print(f"      Output shape: {keypoint_model.output_shape}")
    else:
        keypoint_model = None
        print(f"WARNING: Model file not found at {model_path}, using demo mode")
except Exception as e:
    keypoint_model = None
    print(f"WARNING: Failed to load model: {e}, using demo mode")

def check_sim(i,file_map):
       # Convert input to lowercase for case-insensitive matching
       i_lower = i.lower()
       for item in file_map:
              for word in file_map[item]:
                     if(i_lower==word.lower()):
                            return 1,item
       return -1,""


# Load text-to-sign data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
op_dest = os.path.join(BASE_DIR, "filtered_data")
alpha_dest = os.path.join(BASE_DIR, "alphabet") + os.sep

# Walk through subdirectories to find all .webp files and build a lookup map
file_map = {}
for root, dirs, files in os.walk(op_dest):
    for file in files:
        if file.lower().endswith(".webp"):
            rel_path = os.path.relpath(os.path.join(root, file), op_dest)
            # split the filename (without extension) into words for matching
            name = os.path.splitext(file)[0]
            tokens = name.split()
            file_map[rel_path.replace("\\", "/")] = tokens  # normalize to forward slashes

# debug info
print(f"DEBUG: total webp entries in map: {len(file_map)}")
# show any entries containing the word 'food' (case-insensitive)
for k in file_map:
    if 'food' in k.lower():
        print(f"DEBUG entry for food: {k} -> {file_map[k]}")

# total duration for the entire translation animation (milliseconds)
# changed to 3000ms (3 seconds)
TOTAL_DURATION_MS = 3000

def func(a):
    all_frames=[]
    words=a.split()

    for word in words:
        flag,sim=check_sim(word,file_map)

        # If word NOT found → spell letters
        if(flag==-1):
            for letter in word:
                file_path = os.path.join(alpha_dest, str(letter).lower()+"_small.gif")

                if not os.path.exists(file_path):
                    continue

                im = PIL.Image.open(file_path)

                frames=[]
                if getattr(im,"is_animated",False):
                    for f in range(im.n_frames):
                        im.seek(f)
                        frames.append(im.copy())
                else:
                    frames.append(im.copy())

                for frame_img in frames:
                    frame_img = frame_img.resize((380,260))
                    all_frames.append(frame_img)

        # If full word found (.webp)
        else:
            file_path = os.path.join(op_dest, sim)

            if not os.path.exists(file_path):
                continue

            im = PIL.Image.open(file_path)

            frames=[]
            if getattr(im,"is_animated",False):
                for f in range(im.n_frames):
                    im.seek(f)
                    frames.append(im.copy())
            else:
                frames.append(im.copy())

            for frame_img in frames:
                frame_img = frame_img.resize((380,260))
                all_frames.append(frame_img)

    return all_frames

def translate_words(words):
    # combine frames for all words into a single list
    all_frames = []
    for word in words:
        frames = func(word)
        if frames:
            all_frames.extend(frames)
            # small pause between words
            all_frames.extend([frames[-1]] * 5)
    if all_frames:
        show_frames(all_frames, None)


def show_frames(frames, callback, i=0):
    # render each frame such that total time ~= TOTAL_DURATION_MS
    if i < len(frames):
        display_frame(frames[i])
        delay = TOTAL_DURATION_MS // max(len(frames), 1)
        root.after(delay, lambda: show_frames(frames, callback, i+1))
    else:
        if callback:
            callback()


def display_frame(frame):
    """Display frame in the animated label"""
    if hasattr(display_frame, 'current_label'):
        imgtk = ImageTk.PhotoImage(frame)
        display_frame.current_label.imgtk = imgtk
        display_frame.current_label.configure(image=imgtk)


class Tk_Manage(tk.Tk):
       def __init__(self, *args, **kwargs):     
              tk.Tk.__init__(self, *args, **kwargs)
              container = tk.Frame(self)
              container.pack(side="top", fill="both", expand = True)
              container.grid_rowconfigure(0, weight=1)
              container.grid_columnconfigure(0, weight=1)
              self.frames = {}
              for F in (StartPage, VtoS, StoV):
                     frame = F(container, self)
                     self.frames[F] = frame
                     frame.grid(row=0, column=0, sticky="nsew")
              self.show_frame(StartPage)

       def show_frame(self, cont):
              frame = self.frames[cont]
              frame.tkraise()

        
class StartPage(tk.Frame):

       def __init__(self, parent, controller):
              tk.Frame.__init__(self,parent)
              label = tk.Label(self, text="Two Way Sign Langage Translator", font=("Verdana", 12))
              label.pack(pady=10,padx=10)
              button = tk.Button(self, text="Text to Sign",command=lambda: controller.show_frame(VtoS))
              button.pack()
              button2 = tk.Button(self, text="Sign to Text",command=lambda: controller.show_frame(StoV))
              button2.pack()
              image_path = os.path.join(BASE_DIR, "Two Way Sign Language Translator.png")
              load = PIL.Image.open(image_path)
              load = load.resize((620, 450))
              render = ImageTk.PhotoImage(load)
              img = Label(self, image=render)
              img.image = render
              img.pack(pady=40)
              


class VtoS(tk.Frame):
       def __init__(self, parent, controller):
              cnt=0
              gif_frames=[]
              inputtxt=None
              tk.Frame.__init__(self, parent)
              label = tk.Label(self, text="Text to Sign", font=("Verdana", 12))
              label.pack(pady=10,padx=10)
              gif_box = tk.Label(self)
              
              button1 = tk.Button(self, text="Back to Home",command=lambda: controller.show_frame(StartPage))
              button1.pack()
              button2 = tk.Button(self, text="Sign to Text",command=lambda: controller.show_frame(StoV))
              button2.pack()
              # playback delay will be calculated to fit frames into TOTAL_DURATION_MS
              def gif_stream(delay_ms=None):
                     nonlocal cnt, gif_frames
                     if cnt >= len(gif_frames):
                            return
                     img = gif_frames[cnt]
                     cnt += 1
                     imgtk = ImageTk.PhotoImage(image=img)
                     gif_box.imgtk = imgtk
                     gif_box.configure(image=imgtk)
                     # use computed delay if provided (first call sets it)
                     if delay_ms is None:
                            # compute once based on total frame count
                            frame_count = max(len(gif_frames), 1)
                            delay_ms = TOTAL_DURATION_MS // frame_count
                     gif_box.after(delay_ms, lambda: gif_stream(delay_ms))
              
              def Take_input():
                     INPUT = inputtxt.get("1.0", "end-1c")
                     print(INPUT)
                     nonlocal gif_frames, cnt
                     gif_frames = []
                     cnt = 0
                     words = INPUT.strip().split()
                     # aggregate frames for all words
                     for w in words:
                            gif_frames.extend(func(w))
                     if gif_frames:
                            delay = TOTAL_DURATION_MS // len(gif_frames)
                     else:
                            delay = 100
                     gif_stream(delay)
                     gif_box.place(x=400,y=160)
              
              l = tk.Label(self,text = "Enter Text:")
              inputtxt = tk.Text(self, height = 4,width = 25)
              Display = tk.Button(self, height = 2,width = 20,text ="Convert",command = lambda:Take_input())
              l.place(x=50, y=160)
              inputtxt.place(x=50, y=200)
              Display.pack()


class StoV(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent, bg="white")

        self.controller = controller
        self.cap = None
        self.running = False

        self.current_character = ""
        self.current_word = ""
        self.full_sentence = ""

        self.last_added_char = ""
        self.no_hand_frames = 0
        self.no_hand_threshold = 30  # frames without hand to auto-complete word
        
        # ======= PREDICTION PIPELINE - REAL-TIME =======
        # Confidence threshold: 0.6 (only accept confident predictions)
        # Display: Immediate (no waiting for multiple frames)
        # Duplicates: Prevented (same char won't repeat)
        # This ensures fast, readable text output
        self.confidence_threshold = 0.6  # Accept > 0.6 confidence
        self.stability_buffer_size = 3   # Small buffer for smoothing
        self.stability_buffer = deque(maxlen=self.stability_buffer_size)
        self.last_stable_prediction = None
        self.last_character_time = 0
        self.character_cooldown = 0.1  # 100ms between character additions
        self.space_label = "SPACE"  # Special label for space gesture

        # Use the pre-loaded keypoint model and label map; if missing, predictions are skipped
        self.keypoint_model = keypoint_model
        self.idx_to_label = idx_to_label
        
        # Initialize MediaPipe Hands later (lazy initialization)
        self.mp_hands = None
        
        # ======= WORD CORRECTION DICTIONARY =======
        # Common sign language words and their standard spellings
        self.valid_words = {
            'HELLO': 0, 'HI': 0, 'GOODBYE': 0, 'BYE': 0,
            'THANK': 0, 'THANKS': 0, 'THANK YOU': 0,
            'PLEASE': 0, 'YES': 0, 'NO': 0,
            'FRIEND': 0, 'HELP': 0, 'LOVE': 0, 'HAPPY': 0,
            'NAME': 0, 'SORRY': 0, 'EXCUSE': 0, 'UNDERSTAND': 0,
            'GOOD': 0, 'BAD': 0, 'BEAUTIFUL': 0, 'NICE': 0,
            'WATER': 0, 'FOOD': 0, 'DRINK': 0, 'EAT': 0,
            'YES': 0, 'NO': 0, 'STOP': 0, 'GO': 0,
            'COME': 0, 'GIVE': 0, 'TAKE': 0, 'HELP': 0,
            'LIKE': 0, 'WANT': 0, 'NEED': 0, 'HAVE': 0,
            'TIME': 0, 'DAY': 0, 'NIGHT': 0, 'MORNING': 0,
            'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0,
            'G': 0, 'H': 0, 'I': 0, 'J': 0, 'K': 0, 'L': 0,
        }

        # Configure grid weights for responsive layout
        self.grid_rowconfigure(1, weight=1)  # Give weight to camera display row
        self.grid_columnconfigure(0, weight=1)

        # ================= TITLE =================
        title = tk.Label(self,
                         text="SIGN TO TEXT TRANSLATION",
                         font=("Arial", 30, "bold"),
                         bg="white")
        title.grid(row=0, column=0, sticky="ew", pady=10, padx=20)

        # ================= CAMERA CONTAINER =================
        camera_container = tk.Frame(self, bg="white")
        camera_container.grid(row=1, column=0, sticky="nsew", pady=10, padx=20)
        camera_container.grid_rowconfigure(0, weight=1)
        camera_container.grid_columnconfigure(0, weight=1)
        camera_container.grid_columnconfigure(1, weight=1)

        # LEFT SIDE (Camera Display)
        left_frame = tk.Frame(camera_container, bg="white")
        left_frame.grid(row=0, column=0, padx=10, sticky="nsew")
        left_frame.grid_rowconfigure(1, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        tk.Label(left_frame,
                 text="Camera Display",
                 font=("Arial", 16, "bold"),
                 bg="white").grid(row=0, column=0, sticky="ew")

        self.video_label = tk.Label(left_frame,
                                    bg="black")
        self.video_label.grid(row=1, column=0, sticky="nsew")

        # RIGHT SIDE (Processed Hand)
        right_frame = tk.Frame(camera_container, bg="white")
        right_frame.grid(row=0, column=1, padx=10, sticky="nsew")
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        tk.Label(right_frame,
                 text="Processed Hand",
                 font=("Arial", 16, "bold"),
                 bg="white").grid(row=0, column=0, sticky="ew")

        self.process_label = tk.Label(right_frame,
                                      bg="gray30")
        self.process_label.grid(row=1, column=0, sticky="nsew")

        # ================= TEXT OUTPUT =================
        text_frame = tk.Frame(self, bg="white")
        text_frame.grid(row=2, column=0, sticky="ew", pady=10, padx=20)

        self.char_var = tk.StringVar()
        self.word_var = tk.StringVar()
        self.sentence_var = tk.StringVar()

        tk.Label(text_frame,
                 text="Character :",
                 font=("Arial", 14, "bold"),
                 bg="white").pack(anchor="w", pady=3)

        tk.Label(text_frame,
                 textvariable=self.char_var,
                 font=("Arial", 12),
                 bg="white",
                 fg="#0066cc").pack(anchor="w", pady=3)

        tk.Label(text_frame,
                 text="Word :",
                 font=("Arial", 14, "bold"),
                 bg="white").pack(anchor="w", pady=3)

        tk.Label(text_frame,
                 textvariable=self.word_var,
                 font=("Arial", 12),
                 bg="white",
                 fg="#0066cc").pack(anchor="w", pady=3)

        tk.Label(text_frame,
                 text="Sentence :",
                 font=("Arial", 14, "bold"),
                 bg="white").pack(anchor="w", pady=3)

        tk.Label(text_frame,
                 textvariable=self.sentence_var,
                 font=("Arial", 12),
                 bg="white",
                 fg="#0066cc").pack(anchor="w", pady=3)

        # ================= BUTTONS =================
        button_frame = tk.Frame(self, bg="white")
        button_frame.grid(row=3, column=0, sticky="ew", pady=15, padx=20)
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        button_frame.grid_columnconfigure(2, weight=1)
        button_frame.grid_columnconfigure(3, weight=1)

        start_btn = tk.Button(button_frame,
                              text="Start Camera",
                              font=("Arial", 12, "bold"),
                              height=2,
                              bg="#4CAF50",
                              fg="white",
                              command=self.start_camera)
        start_btn.grid(row=0, column=0, padx=5, sticky="ew")

        stop_btn = tk.Button(button_frame,
                             text="Stop Camera",
                             font=("Arial", 12, "bold"),
                             height=2,
                             bg="#f44336",
                             fg="white",
                             command=self.stop_camera)
        stop_btn.grid(row=0, column=1, padx=5, sticky="ew")

        clear_btn = tk.Button(button_frame,
                              text="Clear Text",
                              font=("Arial", 12, "bold"),
                              height=2,
                              bg="#2196F3",
                              fg="white",
                              command=self.clear_text)
        clear_btn.grid(row=0, column=2, padx=5, sticky="ew")


        back_btn = tk.Button(button_frame,
                             text="Back",
                             font=("Arial", 12, "bold"),
                             height=2,
                             bg="#FF9800",
                             fg="white",
                             command=lambda: controller.show_frame(StartPage))
        back_btn.grid(row=0, column=3, padx=5, sticky="ew")

    # ================= START CAMERA =================
    def start_camera(self):
        """Start the camera and begin frame capture"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open camera")
                return
            # Set camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.running = True
        self.update_frame()

    # ================= STOP CAMERA =================
    def stop_camera(self):
        """Stop camera capture and release resources"""
        self.running = False

        if self.cap:
            self.cap.release()
            self.cap = None

        self.video_label.config(image="")
        self.process_label.config(image="")
        self.video_label.image = None
        self.process_label.image = None

    # ================= CLEAR TEXT =================
    def clear_text(self):

        """Clear all text fields"""

        self.char_var.set("")
        self.word_var.set("")
        self.sentence_var.set("")

        self.current_character = ""
        self.current_word = ""
        self.full_sentence = ""

        self.last_added_char = ""

    def preprocess_frame(self, frame):
        """Extract hand landmarks and return a normalized feature vector plus a display image."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Initialize MediaPipe Hands on first use (lazy initialization)
        if self.mp_hands is None:
            try:
                self.mp_hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7,
                )
            except Exception as e:
                print(f"Error initializing MediaPipe Hands: {e}")
                return None, frame

        try:
            results = self.mp_hands.process(frame_rgb)
        except Exception as e:
            print(f"Error processing frame with MediaPipe: {e}")
            return None, frame

        if not results.multi_hand_landmarks:
            return None, frame

        hand_landmarks = results.multi_hand_landmarks[0]

        h, w, _ = frame.shape
        x_list = [int(lm.x * w) for lm in hand_landmarks.landmark]
        y_list = [int(lm.y * h) for lm in hand_landmarks.landmark]

        x_min, x_max = min(x_list), max(x_list)
        y_min, y_max = min(y_list), max(y_list)

        padding = 30
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        hand_img = frame[y_min:y_max, x_min:x_max]

        # Build normalized feature vector from landmarks
        landmark_list = calc_landmark_list(frame, hand_landmarks)
        processed = pre_process_landmark(landmark_list)

        return processed, hand_img

    def _get_prediction(self, hand_image):
        """
        Get top 3 predictions from model or generate demo prediction.
        Returns: (predicted_label, confidence, runner_up_conf, is_valid)
        
        This method enables ambiguity detection by returning multiple top predictions.
        """
        if self.keypoint_model is None:
            # ===== DEMO MODE =====
            import random
            import time
            
            # Create "sticky" demo predictions that change every 1.5 seconds
            if not hasattr(self, '_demo_prediction_time'):
                self._demo_prediction_time = time.time()
                self._demo_label = random.choice(list(self.idx_to_label.values())[:10])
                self._demo_conf = random.uniform(0.80, 0.95)  # Higher confidence for demo
            
            current_time = time.time()
            if current_time - self._demo_prediction_time > 1.5:
                self._demo_prediction_time = current_time
                self._demo_label = random.choice(list(self.idx_to_label.values())[:10])
                self._demo_conf = random.uniform(0.80, 0.95)
            
            runner_up_conf = self._demo_conf * random.uniform(0.6, 0.8)  # Simulate uncertainty
            return self._demo_label, self._demo_conf, runner_up_conf, True

        # ===== REAL MODEL PREDICTION =====
        try:
            if hand_image is None or hand_image.size == 0:
                return None, 0.0, 0.0, False

            if hand_image.shape[0] < 10 or hand_image.shape[1] < 10:
                return None, 0.0, 0.0, False
            
            # Resize to expected input size
            hand_img_resized = cv2.resize(hand_image, (128, 128))
            
            # Convert to grayscale if needed
            if len(hand_img_resized.shape) == 3:
                hand_img_gray = cv2.cvtColor(hand_img_resized, cv2.COLOR_BGR2GRAY)
            else:
                hand_img_gray = hand_img_resized
            
            # Normalize to [0, 1]
            hand_img_norm = hand_img_gray.astype(np.float32) / 255.0
            
            # Prepare input: (batch, height, width, channels)
            input_frame = np.expand_dims(np.expand_dims(hand_img_norm, axis=0), axis=-1)
            
            # Get prediction - get all probabilities for ambiguity detection
            prediction = self.keypoint_model.predict(input_frame, verbose=0)[0]
            
            # Get top 2 predictions for ambiguity detection
            top_indices = np.argsort(prediction)[-2:][::-1]  # Top 2 in descending order
            top_idx = int(top_indices[0])
            runner_up_idx = int(top_indices[1])
            
            top_conf = float(prediction[top_idx])
            runner_up_conf = float(prediction[runner_up_idx])
            
            predicted_label = self.idx_to_label.get(top_idx, chr(ord('A') + top_idx))
            
            return predicted_label, top_conf, runner_up_conf, True

        except Exception as e:
            print(f"[ERROR] Model prediction failed: {e}")
            return None, 0.0, 0.0, False

    def _levenshtein_distance(self, s1, s2):
        """Calculate Levenshtein distance between two strings (for word correction)"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def _correct_word(self, word):
        """
        Auto-correct word using Levenshtein distance.
        Returns closest valid word if distance ≤ 2, otherwise returns original.
        """
        if not word or word in self.valid_words:
            return word
        
        word_upper = word.upper()
        if word_upper in self.valid_words:
            return word_upper
        
        # Find closest match with distance ≤ 2 edits
        best_match = word_upper
        best_distance = float('inf')
        
        for valid_word in self.valid_words.keys():
            distance = self._levenshtein_distance(word_upper, valid_word)
            if distance < best_distance and distance <= 2:
                best_distance = distance
                best_match = valid_word
        
        if best_distance <= 2:
            print(f"   [CORRECTED] '{word_upper}' → '{best_match}' (distance={best_distance})")
            return best_match
        
        return word_upper

    def _check_ambiguity(self, top_conf, runner_up_conf):
        """
        Detect ambiguity when top 2 predictions are too close.
        Returns: (is_ambiguous, confidence_ratio)
        
        Example: If top=0.85 and runner_up=0.80, ratio=0.941 (ambiguous)
        If top=0.85 and runner_up=0.60, ratio=0.706 (clear)
        """
        if runner_up_conf == 0:
            return False, 1.0
        
        confidence_ratio = runner_up_conf / top_conf
        # Consider ambiguous if runner-up is >70% of top confidence
        is_ambiguous = confidence_ratio > 0.70
        
        return is_ambiguous, confidence_ratio

    def _process_prediction(self, predicted_label, top_conf, runner_up_conf):
        """
        Real-time prediction pipeline.
        SPEC: confidence > 0.6, immediate display, no duplicates.
        """
        # ===== CONFIDENCE CHECK =====
        if top_conf <= self.confidence_threshold:
            print(f"    [REJECTED] conf={top_conf:.3f} <= {self.confidence_threshold}")
            return

        print(f"    [ACCEPTED] conf={top_conf:.3f} > {self.confidence_threshold}")

        # ===== SPACE GESTURE (Complete Word) =====
        if predicted_label == self.space_label:
            if self.current_word.strip():
                corrected = self._correct_word(self.current_word)
                self.full_sentence += corrected + " "
                self.sentence_var.set(self.full_sentence.strip())
                print(f"[SPACE] Word '{corrected}' added to sentence")
                self.current_word = ""
                self.word_var.set("")
                self.char_var.set("")
                self.last_added_char = ""
            return

        # ===== DISPLAY CHARACTER IMMEDIATELY =====
        self.char_var.set(predicted_label)
        print(f"  [CHAR] {predicted_label}")

        # ===== ADD TO WORD (if different from last character) =====
        current_time_ts = datetime.now().timestamp()
        time_since_last_char = current_time_ts - self.last_character_time

        # Only add if:
        # 1. Different from previous character (prevent duplicates)
        # 2. Cooldown time has passed
        if predicted_label != self.last_added_char and time_since_last_char >= self.character_cooldown:
            self.current_word += predicted_label
            self.word_var.set(self.current_word)
            self.last_added_char = predicted_label
            self.last_character_time = current_time_ts
            print(f"  [WORD] '{self.current_word}'")

    def predict_character(self, processed_frame, hand_image=None):
        """
        Get prediction from model and process it.
        """
        predicted_label, top_conf, runner_up_conf, is_valid = self._get_prediction(hand_image)

        if not is_valid or predicted_label is None:
            return

        print(f"[PRED] {predicted_label} @ conf={top_conf:.3f}")
        self._process_prediction(predicted_label, top_conf, runner_up_conf)

    # ================= CAMERA LOOP =================
    def update_frame(self):
        """Continuously update camera frames"""
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()

        if ret:
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)

            # Display camera feed
            self.display_frame(frame, self.video_label)

            # Preprocess and display hand
            processed_frame, mask = self.preprocess_frame(frame)
            self.display_frame(mask, self.process_label)

            if processed_frame is not None:
                # Hand detected - process prediction
                self.no_hand_frames = 0
                self.predict_character(processed_frame, hand_image=mask)
            else:
                # No hand detected
                self.no_hand_frames += 1
                
                if self.no_hand_frames == 1:
                    print("[NO HAND]")

        self.after(30, self.update_frame)



    # ================= DISPLAY FRAME =================
    def display_frame(self, frame, label):
        """Convert OpenCV frame to PhotoImage and display"""
        # Handle grayscale to RGB conversion
        if len(frame.shape) == 2:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to fit display panels
        frame_rgb = cv2.resize(frame_rgb, (700, 520))

        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image=pil_image)

        # Update label with new image
        label.config(image=photo)
        label.image = photo  # Keep reference to prevent garbage collection

if __name__ == "__main__":
    app = Tk_Manage()
    app.geometry("1600x900")
    app.mainloop()