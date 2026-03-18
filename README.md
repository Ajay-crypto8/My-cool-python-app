# My-cool-python-app
First python coding app 
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDFlatButton

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3
import time
from gtts import gTTS   # better for Indian languages if needed
import playsound
import os

class SignDetector(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'

        self.lbl_status = MDLabel(text="Starting camera...", halign="center", theme_text_color="Primary")
        self.img = Image(size_hint=(1, 0.8))
        self.lbl_result = MDLabel(text="Sign: Waiting...", halign="center", font_style="H5")

        btn_quit = MDFlatButton(text="Quit", on_press=self.stop)
        
        self.add_widget(self.lbl_status)
        self.add_widget(self.img)
        self.add_widget(self.lbl_result)
        self.add_widget(btn_quit)

        # ── ML & MediaPipe setup ──
        self.labels = [line.strip() for line in open("labels.txt", 'r').readlines()]
        
        self.interpreter = tf.lite.Interpreter(model_path="model.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.IMG_SIZE = 224

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)
        self.mp_draw = mp.solutions.drawing_utils

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.lbl_status.text = "Error: Cannot open camera!"
            return

        self.engine = pyttsx3.init()
        self.last_spoken = ""
        self.last_time = 0
        self.cooldown = 1.8  # seconds

        Clock.schedule_interval(self.update, 1.0 / 30.0)  # \~30 fps

    def update(self, dt):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        sign_text = "No sign"
        conf_str = ""

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, handLms, self.mp_hands.HAND_CONNECTIONS)

                # Crop hand (simple bbox)
                h, w, _ = frame.shape
                x_list = [lm.x * w for lm in handLms.landmark]
                y_list = [lm.y * h for lm in handLms.landmark]
                x1, x2 = int(min(x_list)), int(max(x_list))
                y1, y2 = int(min(y_list)), int(max(y_list))

                pad = 40
                x1, x2 = max(0, x1-pad), min(w, x2+pad)
                y1, y2 = max(0, y1-pad), min(h, y2+pad)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                resized = cv2.resize(crop, (self.IMG_SIZE, self.IMG_SIZE))
                input_img = np.expand_dims(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0, 0)

                self.interpreter.set_tensor(self.input_details[0]['index'], input_img)
                self.interpreter.invoke()
                preds = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

                idx = np.argmax(preds)
                conf = preds[idx]

                if conf > 0.75:
                    label = self.labels[idx]
                    sign_text = label
                    conf_str = f" {conf:.0%}"

                    now = time.time()
                    if label != self.last_spoken or (now - self.last_time > self.cooldown):
                        # Option 1: pyttsx3 (offline)
                        # self.engine.say(label)
                        # self.engine.runAndWait()

                        # Option 2: gTTS (online, better voices, supports Tamil/English)
                        tts = gTTS(text=label, lang='en')   # change to 'ta' for Tamil if labels in Tamil
                        tts.save("temp.mp3")
                        playsound.playsound("temp.mp3")
                        os.remove("temp.mp3")

                        self.last_spoken = label
                        self.last_time = now

        # Show on Kivy image
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img.texture = texture

        self.lbl_result.text = f"Sign: {sign_text}{conf_str}"

    def stop(self, *args):
        self.cap.release()
        App.get_running_app().stop()

class SignApp(App):
    def build(self):
        return SignDetector()

if __name__ == '__main__':
    SignApp().run()
